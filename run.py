"""Pipeline: detect + track cầu thủ (YOLO + ByteTrack) -> nhận dạng số áo theo tracklet.

    python run.py --source video.mp4 --output outputs/

Các bước, mỗi frame:
  1. YOLO detect người và ByteTrack gán track id.
  2. Mỗi FRAME_STRIDE frame, với từng cầu thủ đủ lớn:
     a. Legibility classifier (ResNet34) trên crop cầu thủ: có thấy số áo không?
     b. Lấy keypoints vai/hông (từ YOLO-pose) rồi cắt vùng thân áo.
     c. PARSeq đọc số trên crop thân áo.
  3. Hết video: gộp các lần đọc theo track (bỏ phiếu có trọng số) để ra 1 số áo / track.
  4. Render video kết quả và xuất JSON/CSV.
"""
import argparse
import csv
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import config
from pipeline.aggregator import TrackletVoter
from pipeline.device import pick_device
from pipeline.legibility import LegibilityModel
from pipeline.profiler import StepTimer
from pipeline.str_recognizer import JerseyNumberRecognizer
from pipeline.torso import match_pose_to_box, torso_box

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video, ảnh, hoặc folder chứa video/ảnh")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--models-dir", default=str(config.MODELS_DIR))
    ap.add_argument("--det-model", default=config.DET_MODEL)
    ap.add_argument("--pose-model", default=config.POSE_MODEL)
    ap.add_argument("--classes", type=int, nargs="+", default=config.PLAYER_CLASSES,
                    help="class id của cầu thủ trong detector (COCO person = 0)")
    ap.add_argument("--tracker", default=config.TRACKER, help="bytetrack.yaml | botsort.yaml")
    ap.add_argument("--conf", type=float, default=config.DET_CONF)
    ap.add_argument("--imgsz", type=int, default=config.DET_IMGSZ)
    ap.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    ap.add_argument("--min-box-height", type=int, default=config.MIN_BOX_HEIGHT)
    ap.add_argument("--legibility-th", type=float, default=config.LEGIBILITY_THRESHOLD)
    ap.add_argument("--no-legibility", action="store_true",
                    help="bỏ qua legibility classifier, đọc số trên mọi cầu thủ (chậm hơn, nhiều nhiễu hơn)")
    ap.add_argument("--kpt-conf", type=float, default=config.KPT_CONF)
    ap.add_argument("--max-reads", type=int, default=config.MAX_READS_PER_TRACK)
    ap.add_argument("--sum-th", type=float, default=config.SUM_THRESHOLD)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = toàn bộ video")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--no-video", action="store_true", help="không render video kết quả")
    ap.add_argument("--save-crops", action="store_true", help="lưu crop thân áo + số đọc được (debug)")
    return ap.parse_args()


class PlayerJerseyPipeline:
    def __init__(self, args):
        self.args = args
        self.device = pick_device(args.device)
        models_dir = Path(args.models_dir)
        print(f"[init] device={self.device}")
        self.legibility = LegibilityModel(models_dir / config.LEGIBILITY_FILE, self.device, args.legibility_th)
        self.recognizer = JerseyNumberRecognizer(models_dir / config.STR_FILE, self.device)
        self.det_is_pose = "pose" in Path(args.det_model).stem
        self.pose = None if self.det_is_pose else YOLO(args.pose_model)

    # ------------------------------------------------------------------ pass 1
    def analyse(self, frames, total, prof: StepTimer, crops_dir=None):
        a = self.args
        detector = YOLO(a.det_model)  # model mới cho mỗi video -> reset tracker
        yolo_kw = dict(conf=a.conf, imgsz=a.imgsz, classes=a.classes, device=self.device, verbose=False)
        voter = TrackletVoter(config.FILTER_THRESHOLD, a.sum_th, config.USE_BIAS)
        frame_log = []            # [(frame_idx, track_id, x1, y1, x2, y2, det_conf)]
        span = {}                 # track_id -> [first_frame, last_frame]
        stats = defaultdict(lambda: {"attempts": 0, "legible": 0, "no_pose": 0})

        for idx, frame in tqdm(timed_frames(frames, prof), total=total, desc="analyse"):
            with prof("2. detect+track", n_items=1):
                t0 = time.perf_counter()
                r = detector.track(frame, persist=True, tracker=a.tracker, **yolo_kw)[0]
                total_ms = (time.perf_counter() - t0) * 1000
            sp = r.speed  # ms, ultralytics tự đo
            prof.add("   ├ yolo preprocess", sp.get("preprocess", 0))
            prof.add("   ├ yolo inference", sp.get("inference", 0))
            prof.add("   ├ yolo postprocess", sp.get("postprocess", 0))
            prof.add("   └ tracker (ByteTrack)", max(0.0, total_ms - sum(sp.get(k, 0) for k in
                                                                    ("preprocess", "inference", "postprocess"))))
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kpts = r.keypoints.data.cpu().numpy() if self.det_is_pose and r.keypoints is not None else None

            for b, t, c in zip(boxes, ids, confs):
                frame_log.append((idx, int(t), *[int(v) for v in b], float(c)))
                span.setdefault(int(t), [idx, idx])[1] = idx

            if idx % a.stride:
                continue

            # --- chọn cầu thủ cần đọc số ở frame này
            cand = [i for i, (b, t) in enumerate(zip(boxes, ids))
                    if b[3] - b[1] >= a.min_box_height and stats[int(t)]["attempts"] < a.max_reads]
            if not cand:
                continue
            if a.no_legibility:
                leg_prob = np.ones(len(cand))
            else:
                with prof("3. legibility", n_items=len(cand)):
                    player_crops = [frame[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2])]
                                    for i in cand]
                    leg_prob = self.legibility.predict(player_crops)

            legible = [i for i, p in zip(cand, leg_prob) if p > a.legibility_th]
            for i in cand:
                stats[int(ids[i])]["attempts"] += 1
            if not legible:
                continue

            # --- keypoints -> crop thân áo
            if kpts is None:
                with prof("4. pose (model riêng)", n_items=1):
                    pr = self.pose(frame, imgsz=a.imgsz, device=self.device, verbose=False)[0]
                    p_boxes = pr.boxes.xyxy.cpu().numpy() if pr.boxes is not None else np.zeros((0, 4))
                    p_kpts = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None else None
            torso_crops, owners = [], []
            t_crop = time.perf_counter()
            for i in legible:
                tid = int(ids[i])
                stats[tid]["legible"] += 1
                if kpts is not None:
                    k = kpts[i]
                else:
                    j = match_pose_to_box(boxes[i], p_boxes)
                    k = p_kpts[j] if j >= 0 and p_kpts is not None else None
                tb = torso_box(k, frame.shape, a.kpt_conf, config.TORSO_PADDING)
                if tb is None:
                    stats[tid]["no_pose"] += 1
                    continue
                x1, y1, x2, y2 = tb
                torso_crops.append(frame[y1:y2, x1:x2])
                owners.append(tid)
            prof.add("5. torso crop", (time.perf_counter() - t_crop) * 1000, n_items=len(legible))

            # --- PARSeq đọc số
            if not torso_crops:
                continue
            with prof("6. STR (PARSeq)", n_items=len(torso_crops)):
                preds = self.recognizer.predict(torso_crops)
            for tid, crop, (label, prob) in zip(owners, torso_crops, preds):
                voter.add(tid, label, prob)
                if crops_dir is not None:
                    with prof("   save crops (debug)", n_items=1):
                        cv2.imwrite(str(crops_dir / f"t{tid:04d}_f{idx:06d}_{label or 'x'}_{prob:.2f}.jpg"), crop)

        return voter, frame_log, span, stats

    # ------------------------------------------------------------------ pass 2
    @staticmethod
    def render(src, dst, frame_log, numbers, fps, size, max_frames, prof: StepTimer):
        by_frame = defaultdict(list)
        for row in frame_log:
            by_frame[row[0]].append(row)
        tmp = dst.with_suffix(".tmp.mp4")
        t0 = time.perf_counter()
        n = 0
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for idx, frame in tqdm(read_frames(src, max_frames), total=None, desc="render"):
            for _, tid, x1, y1, x2, y2, _ in by_frame.get(idx, []):
                num = numbers.get(tid, -1)
                color = (0, 220, 0) if num > 0 else (160, 160, 160)
                text = f"#{num}" if num > 0 else f"id{tid}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            writer.write(frame)
            n += 1
        writer.release()
        prof.add("9. render video", (time.perf_counter() - t0) * 1000, n_items=n)
        # mp4v không xem được trong trình duyệt/notebook -> chuyển sang H.264 nếu có ffmpeg
        if shutil.which("ffmpeg"):
            with prof("10. ffmpeg -> H.264", n_items=1):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-preset", "fast", str(dst)], check=False)
            if dst.exists():
                tmp.unlink()
                return dst
        tmp.rename(dst)
        return dst

    # ------------------------------------------------------------------ run
    def process_video(self, src: Path, out_root: Path):
        a = self.args
        out_dir = out_root / src.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = None
        if a.save_crops:
            crops_dir = out_dir / "torso_crops"
            crops_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if a.max_frames:
            total = min(total or a.max_frames, a.max_frames)

        prof = StepTimer(self.device)
        t_start = time.perf_counter()
        voter, frame_log, span, stats = self.analyse(read_frames(src, a.max_frames), total, prof, crops_dir)
        elapsed = time.perf_counter() - t_start

        t_agg = time.perf_counter()
        tracks, numbers = {}, {}
        for tid in sorted(span):
            num, weight = voter.best(tid)
            numbers[tid] = num
            tracks[tid] = {
                "jersey_number": num,
                "weight": round(weight, 3),
                "votes": {str(k): round(v, 3) for k, v in sorted(voter.weights(tid).items(), key=lambda kv: -kv[1])},
                "first_frame": span[tid][0],
                "last_frame": span[tid][1],
                **stats[tid],
                "str_reads": len(voter.reads.get(tid, [])),
            }
        prof.add("7. aggregate (vote)", (time.perf_counter() - t_agg) * 1000, n_items=len(span))

        t_out = time.perf_counter()
        with open(out_dir / "tracks.json", "w") as f:
            json.dump({"source": str(src), "fps": fps, "tracks": tracks}, f, indent=2)
        with open(out_dir / "detections.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "det_conf", "jersey_number"])
            for row in frame_log:
                w.writerow([*row[:6], f"{row[6]:.3f}", numbers.get(row[1], -1)])
        prof.add("8. write json/csv", (time.perf_counter() - t_out) * 1000)

        n_frames = len(prof.calls.get("2. detect+track", []))
        identified = {t: n for t, n in numbers.items() if n > 0}
        print(f"\n[{src.name}] {n_frames} frames, {elapsed:.1f}s ({n_frames / max(elapsed, 1e-6):.1f} fps)")
        print(f"  tracks: {len(tracks)}, có số áo: {len(identified)}")
        for t, n in sorted(identified.items(), key=lambda kv: kv[1]):
            print(f"    track {t:>4}: #{n:<3} (weight {tracks[t]['weight']}, frames {span[t][0]}-{span[t][1]})")

        if not a.no_video:
            path = self.render(src, out_dir / f"{src.stem}_jersey.mp4", frame_log, numbers, fps, size,
                               a.max_frames, prof)
            print(f"  video: {path}")
        print(f"  json:  {out_dir / 'tracks.json'}")

        timing = prof.summary(n_frames, time.perf_counter() - t_start)
        timing["analyse_fps"] = round(n_frames / max(elapsed, 1e-9), 2)
        timing["config"] = {"det_model": a.det_model, "pose_model": None if self.det_is_pose else a.pose_model,
                            "imgsz": a.imgsz, "stride": a.stride, "no_legibility": a.no_legibility}
        StepTimer.print_table(timing)
        print(f"  analyse (pass 1, không tính render): {timing['analyse_fps']} fps")
        with open(out_dir / "timing.json", "w") as f:
            json.dump(timing, f, indent=2, ensure_ascii=False)
        print(f"  timing: {out_dir / 'timing.json'}")


def timed_frames(frames, prof: StepTimer):
    """Bọc generator đọc frame để đo thời gian decode video."""
    it = iter(frames)
    while True:
        t0 = time.perf_counter()
        try:
            item = next(it)
        except StopIteration:
            return
        prof.add("1. decode frame", (time.perf_counter() - t0) * 1000)
        yield item


def read_frames(src: Path, max_frames: int = 0):
    """Yield (frame_idx, frame BGR). Ảnh đơn được coi là video 1 frame."""
    if src.suffix.lower() in IMAGE_EXTS:
        yield 0, cv2.imread(str(src))
        return
    cap = cv2.VideoCapture(str(src))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        yield idx, frame
        idx += 1
    cap.release()


def collect_sources(source: str):
    p = Path(source)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
    else:
        files = [p]
    if not files:
        raise SystemExit(f"Không tìm thấy video/ảnh trong {source}")
    return files


def main():
    args = parse_args()
    pipe = PlayerJerseyPipeline(args)
    out_root = Path(args.output)
    for src in collect_sources(args.source):
        pipe.process_video(src, out_root)


if __name__ == "__main__":
    main()
