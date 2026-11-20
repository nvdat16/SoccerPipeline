"""Chẩn đoán vì sao số áo không được nhận diện.

Khác run.py: chạy MỌI bước trên MỌI cầu thủ (không lọc), rồi ghi lại từng crop:
  box_h, legibility, conf 4 keypoint vai/hông, kích thước torso, PARSeq đọc gì (kể cả khi
  legibility/pose đã loại). Từ đó thấy được crop "đọc được" đang bị chặn ở bước nào.

    python tools/diagnose.py --source video.mp4 --det-model yolo26s-soccer.pt \
        --pose-model yolo26n-pose.pt --every 5 --out outputs/diagnose

Output: attempts.csv, crops/ (player + torso của các crop bị chặn dù PARSeq đọc chắc chắn), summary in ra màn hình.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline.aggregator import is_valid_number  # noqa: E402
from pipeline.device import pick_device  # noqa: E402
from pipeline.legibility import LegibilityModel  # noqa: E402
from pipeline.str_recognizer import JerseyNumberRecognizer  # noqa: E402
from pipeline.torso import TORSO_KPTS, match_pose_to_box, torso_box  # noqa: E402
from run import resolve_classes  # noqa: E402

TORSO_LOOSE_KPT = 0.05   # crop torso "lỏng" để PARSeq vẫn đọc thử khi keypoint yếu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="outputs/diagnose")
    ap.add_argument("--det-model", default=config.DET_MODEL)
    ap.add_argument("--pose-model", default=config.POSE_MODEL)
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--imgsz", type=int, default=config.DET_IMGSZ)
    ap.add_argument("--conf", type=float, default=0.3, help="không có tracker -> cần ngưỡng cao hơn config.DET_CONF")
    ap.add_argument("--every", type=int, default=5, help="lấy mẫu mỗi N frame")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-box-height", type=int, default=0, help="0 = không lọc, để xem cả box nhỏ")
    ap.add_argument("--models-dir", default=str(config.MODELS_DIR))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-limit", type=int, default=150)
    a = ap.parse_args()

    dev = pick_device(a.device)
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    det = YOLO(a.det_model)
    classes = resolve_classes(det.names, a.classes)
    det_is_pose = det.task == "pose"
    pose = None if det_is_pose else YOLO(a.pose_model)
    leg = LegibilityModel(Path(a.models_dir) / config.LEGIBILITY_FILE, dev)
    rec = JerseyNumberRecognizer(Path(a.models_dir) / config.STR_FILE, dev)
    print(f"device={dev} classes={ {i: det.names[i] for i in classes} } pose={'detector' if det_is_pose else a.pose_model}")

    cap = cv2.VideoCapture(a.source)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(n_total, a.start + a.max_frames) if a.max_frames else n_total
    frame_ids = list(range(a.start, end, a.every))

    rows, saved = [], 0
    for fi in tqdm(frame_ids, desc="diagnose"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        r = det(frame, conf=a.conf, imgsz=a.imgsz, classes=classes, device=dev, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        if det_is_pose:
            kpts_all = r.keypoints.data.cpu().numpy()
        else:
            pr = pose(frame, imgsz=a.imgsz, device=dev, verbose=False)[0]
            p_boxes = pr.boxes.xyxy.cpu().numpy() if pr.boxes is not None else np.zeros((0, 4))
            p_kpts = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None and len(p_boxes) else None

        keep = [i for i, b in enumerate(boxes) if b[3] - b[1] >= a.min_box_height]
        if not keep:
            continue
        players = [frame[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2])] for i in keep]
        leg_p = leg.predict(players)

        torsos, meta = [], []
        for n, i in enumerate(keep):
            b = boxes[i]
            if det_is_pose:
                k, iou_ok = kpts_all[i], True
            else:
                j = match_pose_to_box(b, p_boxes)
                iou_ok = j >= 0
                k = p_kpts[j] if iou_ok and p_kpts is not None else None
            kc = k[TORSO_KPTS, 2] if k is not None else np.zeros(4)
            tb_strict = torso_box(k, frame.shape, config.KPT_CONF, config.TORSO_PADDING)
            tb_loose = tb_strict or torso_box(k, frame.shape, TORSO_LOOSE_KPT, config.TORSO_PADDING)
            if tb_loose is None:
                # fallback: lấy 15%-55% chiều cao box làm "thân áo" để PARSeq vẫn đọc thử
                x1, y1, x2, y2 = b.astype(int)
                h = y2 - y1
                tb_loose = (x1, y1 + int(.15 * h), x2, y1 + int(.55 * h))
            x1, y1, x2, y2 = tb_loose
            torsos.append(frame[y1:y2, x1:x2])
            meta.append((n, i, b, kc, iou_ok, tb_strict is not None, tb_loose))
        preds = rec.predict(torsos)

        for (n, i, b, kc, iou_ok, strict_ok, tb), (label, prob) in zip(meta, preds):
            valid = is_valid_number(label)
            legible = leg_p[n] > config.LEGIBILITY_THRESHOLD
            if not legible:
                blocked = "legibility"
            elif not iou_ok:
                blocked = "pose_no_match"
            elif not strict_ok:
                blocked = "pose_low_kpt"
            elif not valid:
                blocked = "str_invalid"
            elif prob < config.FILTER_THRESHOLD:
                blocked = "str_low_prob"
            else:
                blocked = "ok"
            row = dict(frame=fi, box_h=int(b[3] - b[1]), box_w=int(b[2] - b[0]), cls=det.names[int(r.boxes.cls[i])],
                       legibility=round(float(leg_p[n]), 3), pose_match=iou_ok,
                       kpt_min=round(float(kc.min()), 2), kpt_rs=round(float(kc[0]), 2), kpt_ls=round(float(kc[1]), 2),
                       kpt_lh=round(float(kc[2]), 2), kpt_rh=round(float(kc[3]), 2),
                       torso_h=tb[3] - tb[1], torso_w=tb[2] - tb[0],
                       str_label=label, str_prob=round(prob, 3), blocked=blocked)
            rows.append(row)
            # lưu các crop đáng ngờ: PARSeq đọc chắc chắn mà vẫn bị chặn
            if blocked != "ok" and valid and prob > 0.8 and saved < a.save_limit:
                tag = f"f{fi:06d}_h{row['box_h']}_{blocked}_leg{row['legibility']:.2f}_kpt{row['kpt_min']:.2f}_read{label}_{prob:.2f}"
                cv2.imwrite(str(out / "crops" / f"{tag}_player.jpg"), players[n])
                x1, y1, x2, y2 = tb
                cv2.imwrite(str(out / "crops" / f"{tag}_torso.jpg"), frame[y1:y2, x1:x2])
                saved += 1

    if not rows:
        print(f"\nKhông có cầu thủ nào được detect trong {len(frame_ids)} frame đã lấy mẫu "
              f"(frame {a.start}..{end}, mỗi {a.every}). Thử tăng --max-frames / đổi --start / giảm --every.")
        return
    with open(out / "attempts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\n-> {out / 'attempts.csv'}  |  crops bị chặn dù đọc được: {out / 'crops'} ({saved} ảnh)")


def summarize(rows):
    import collections
    n = len(rows)
    print(f"\n=== {n} crop cầu thủ")
    c = collections.Counter(r["blocked"] for r in rows)
    print("Kết quả theo bước (bước đầu tiên chặn crop):")
    for k in ["ok", "legibility", "pose_no_match", "pose_low_kpt", "str_invalid", "str_low_prob"]:
        print(f"  {k:<15}{c.get(k, 0):>7}  ({100 * c.get(k, 0) / max(n, 1):.1f}%)")

    readable = [r for r in rows if is_valid_number(r["str_label"]) and r["str_prob"] > 0.8]
    print(f"\nCrop PARSeq đọc CHẮC CHẮN (số hợp lệ, prob>0.8): {len(readable)}")
    cr = collections.Counter(r["blocked"] for r in readable)
    for k, v in cr.most_common():
        print(f"  {k:<15}{v:>7}  ({100 * v / max(len(readable), 1):.1f}%)")

    def q(vals):
        return np.percentile(vals, [10, 50, 90]).round(2).tolist() if vals else "-"
    blocked_leg = [r for r in readable if r["blocked"] == "legibility"]
    blocked_kpt = [r for r in readable if r["blocked"] == "pose_low_kpt"]
    if blocked_leg:
        print(f"\n  bị legibility chặn: legibility p10/p50/p90 = {q([r['legibility'] for r in blocked_leg])}, "
              f"box_h = {q([r['box_h'] for r in blocked_leg])}")
    if blocked_kpt:
        print(f"  bị keypoint chặn:   kpt_min p10/p50/p90 = {q([r['kpt_min'] for r in blocked_kpt])}, "
              f"box_h = {q([r['box_h'] for r in blocked_kpt])}")
    ok = [r for r in rows if r["blocked"] == "ok"]
    if ok:
        print(f"  crop ok:            box_h p10/p50/p90 = {q([r['box_h'] for r in ok])}")


if __name__ == "__main__":
    main()
