"""Đo chất lượng tracking + định danh của cầu thủ Ở XA (box nhỏ) từ output của run.py.

    python tools/far_report.py outputs/<run>/<video>                  # chỉ đọc CSV, không cần GPU
    python tools/far_report.py outputs/<run>/<video> --video clip.mp4 --recall-frames 40   # + ước lượng recall

Đọc detections.csv (+ reads.csv) và in:
  A. phân bố chiều cao box cầu thủ theo nhóm kích thước;
  B. track bị đứt GIỮA shot (không phải do cắt cảnh / ra mép frame) trên 1000 detection, theo kích thước;
  C. "gặp nhau" (encounter): 2 cầu thủ CÙNG ĐỘI cách nhau < --enc-r mét trên sân -> nơi ByteTrack hay hoán ID;
     sau mỗi lần gặp: track có đứt / đổi đội trong ±N frame không;
  D. track đọc ra >= 2 số áo khác nhau, mỗi số đủ phiếu, ở 2 quãng thời gian tách biệt -> dấu hiệu hoán ID;
     có lần gặp đồng đội nào nằm giữa 2 quãng không;
  E. độ phủ số áo theo kích thước (tự đọc / nhờ nối);
  F. (--video) recall box nhỏ: chạy detector full frame vs. cắt tile, đếm cầu thủ nhỏ CHỈ tile mới thấy.
Kết quả cũng ghi ra far_report.json cạnh detections.csv để so sánh giữa các lần chạy.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKETS = [(0, 30), (30, 45), (45, 60), (60, 100), (100, 10_000)]
PLAYER_CLASSES = {"player", "goalkeeper", "person", "goalie"}


def bucket_of(h):
    for k, (lo, hi) in enumerate(BUCKETS):
        if lo <= h < hi:
            return k
    return len(BUCKETS) - 1


def bname(k):
    lo, hi = BUCKETS[k]
    return f"<{hi}px" if lo == 0 else (f">={lo}px" if hi >= 10_000 else f"{lo}-{hi}px")


def load(run_dir):
    rows = []
    with open(run_dir / "detections.csv") as f:
        for r in csv.DictReader(f):
            if r["class"] not in PLAYER_CLASSES:
                continue
            x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
            rows.append({"f": int(r["frame"]), "t": int(r["track_id"]), "box": (x1, y1, x2, y2), "h": y2 - y1,
                         "team": int(r["team"]), "shot": int(r["shot_id"]),
                         "xy": (float(r["x_m"]), float(r["y_m"])) if r["x_m"] else None,
                         "num": int(r["jersey_number"]), "src": r["number_source"], "pid": r["player_id"],
                         "h_status": r["h_status"]})
    reads = []
    if (run_dir / "reads.csv").exists():
        with open(run_dir / "reads.csv") as f:
            reads = [(int(r["frame"]), int(r["track_id"]), r["label"], float(r["prob"])) for r in csv.DictReader(f)]
    meta = json.loads((run_dir / "tracks.json").read_text()) if (run_dir / "tracks.json").exists() else {}
    return rows, reads, meta


# ---------------------------------------------------------------------------------------------- A + B + E
def size_and_breaks(rows, W, H, edge=12, tail=5):
    by_t = defaultdict(list)
    shot_last = defaultdict(int)
    shot_first = {}
    for r in rows:
        by_t[r["t"]].append(r)
        shot_last[r["shot"]] = max(shot_last[r["shot"]], r["f"])
        shot_first[r["shot"]] = min(shot_first.get(r["shot"], r["f"]), r["f"])
    n_det = np.zeros(len(BUCKETS), int)
    for r in rows:
        n_det[bucket_of(r["h"])] += 1
    ends = {"cut": np.zeros(len(BUCKETS), int), "edge": np.zeros(len(BUCKETS), int),
            "lost": np.zeros(len(BUCKETS), int)}
    lost_events = []                                  # (frame, track, bucket, xy) để ghép với lần gặp nhau
    lens = defaultdict(list)
    for t, rs in by_t.items():
        rs.sort(key=lambda r: r["f"])
        last = rs[-1]
        k = bucket_of(float(np.median([r["h"] for r in rs[-tail:]])))
        lens[k].append(len(rs))
        x1, y1, x2, y2 = last["box"]
        if last["f"] >= shot_last[last["shot"]] - 1:
            ends["cut"][k] += 1
        elif x1 <= edge or y1 <= edge or x2 >= W - edge or y2 >= H - edge:
            ends["edge"][k] += 1
        else:
            ends["lost"][k] += 1
            lost_events.append((last["f"], t, k))
    cov = {s: np.zeros(len(BUCKETS), int) for s in ("read", "linked", "uncertain")}
    for r in rows:
        if r["num"] > 0 and r["src"] in cov:
            cov[r["src"]][bucket_of(r["h"])] += 1
    return n_det, ends, lost_events, lens, cov


# ---------------------------------------------------------------------------------------------- C
def encounters(rows, enc_r, min_gap):
    """Các lần 2 track cùng đội (đội đã biết) cách nhau < enc_r mét. Gộp frame liên tiếp (cách <= min_gap) thành 1 lần."""
    by_f = defaultdict(list)
    for r in rows:
        if r["xy"] is not None and r["team"] >= 0:
            by_f[r["f"]].append(r)
    pair_frames = defaultdict(list)
    for f, rs in by_f.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["team"] != b["team"]:
                    continue
                d = np.hypot(a["xy"][0] - b["xy"][0], a["xy"][1] - b["xy"][1])
                if d < enc_r:
                    key = (min(a["t"], b["t"]), max(a["t"], b["t"]))
                    pair_frames[key].append((f, min(a["h"], b["h"])))
    events = []
    for (ta, tb), fs in pair_frames.items():
        fs.sort()
        start, prev, hs = fs[0][0], fs[0][0], [fs[0][1]]
        for f, h in fs[1:] + [(None, None)]:
            if f is not None and f - prev <= min_gap:
                prev = f
                hs.append(h)
                continue
            events.append({"tracks": (ta, tb), "start": start, "end": prev, "bucket": bucket_of(float(np.median(hs)))})
            if f is not None:
                start, prev, hs = f, f, [h]
    return events


def encounter_outcomes(events, rows, lost_events, window):
    """Sau mỗi lần gặp: trong `window` frame có track nào trong cặp bị đứt giữa shot / đổi đội không."""
    lost_at = defaultdict(list)
    for f, t, _ in lost_events:
        lost_at[t].append(f)
    team_seq = defaultdict(list)
    for r in rows:
        if r["team"] >= 0:
            team_seq[r["t"]].append((r["f"], r["team"]))
    for t in team_seq:
        team_seq[t].sort()

    def team_flip(t, lo, hi):
        seq = [k for f, k in team_seq.get(t, []) if lo <= f <= hi]
        return len(set(seq)) > 1

    out = {"lost": 0, "team_flip": 0}
    for e in events:
        lo, hi = e["start"] - window, e["end"] + window
        e["lost"] = any(lo <= f <= hi for t in e["tracks"] for f in lost_at.get(t, []))
        e["team_flip"] = any(team_flip(t, lo, hi) for t in e["tracks"])
        out["lost"] += e["lost"]
        out["team_flip"] += e["team_flip"]
    return out


# ---------------------------------------------------------------------------------------------- D
def number_conflicts(reads, events, min_prob=0.5, min_weight=1.0):
    """Track có >= 2 số áo, mỗi số tổng xác suất >= min_weight và hai quãng đọc không xen nhau."""
    by_t = defaultdict(lambda: defaultdict(list))
    for f, t, label, p in reads:
        if label and label.isdigit() and 0 < int(label) < 100 and p >= min_prob:
            by_t[t][int(label)].append((f, p))
    enc_by_t = defaultdict(list)
    for e in events:
        for t in e["tracks"]:
            enc_by_t[t].append(e)
    out = []
    for t, nums in by_t.items():
        strong = {n: v for n, v in nums.items() if sum(p for _, p in v) >= min_weight}
        if len(strong) < 2:
            continue
        (n1, v1), (n2, v2) = sorted(strong.items(), key=lambda kv: -sum(p for _, p in kv[1]))[:2]
        f1, f2 = sorted(f for f, _ in v1), sorted(f for f, _ in v2)
        separated = f1[-1] < f2[0] or f2[-1] < f1[0]
        lo, hi = (f1[-1], f2[0]) if f1[-1] < f2[0] else (f2[-1], f1[0])
        enc_between = [e for e in enc_by_t.get(t, []) if separated and e["end"] >= lo and e["start"] <= hi]
        out.append({"track": t, "numbers": [n1, n2], "frames": [[f1[0], f1[-1]], [f2[0], f2[-1]]],
                    "separated": separated, "encounters_between": len(enc_between)})
    return out


# ---------------------------------------------------------------------------------------------- F
def recall_check(video, rows, n_frames, det_model, imgsz, conf, device, tile_grid, overlap, small_h, out_dir):
    """So detector full frame với detector trên tile: cầu thủ nhỏ chỉ tile mới thấy ~ cầu thủ full frame bỏ sót."""
    import cv2
    from ultralytics import YOLO

    import config
    from pipeline.device import pick_device
    from pipeline.tiling import tiled_detect

    wide = sorted({r["f"] for r in rows if r["h_status"] == "kp"})
    if not wide:
        print("  [F] không có frame wide (h_status=kp) -> bỏ qua")
        return None
    pick = set(np.array(wide)[np.linspace(0, len(wide) - 1, min(n_frames, len(wide))).astype(int)].tolist())
    model = YOLO(det_model)
    dev = pick_device(device)
    names = {i: str(n).lower() for i, n in model.names.items()}
    cls = [i for i, n in names.items() if n in PLAYER_CLASSES]
    cap = cv2.VideoCapture(str(video))
    idx, stats = 0, {"full_small": 0, "tile_only_small": 0, "frames": 0}
    crops_dir = out_dir / "far_recall_crops"
    crops_dir.mkdir(exist_ok=True)
    saved = 0
    while pick:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in pick:
            pick.discard(idx)
            full = model.predict(frame, imgsz=imgsz, conf=conf, classes=cls, device=dev, verbose=False)[0]
            fb = full.boxes.xyxy.cpu().numpy() if full.boxes is not None else np.zeros((0, 4))
            tb, tc, _, _ = tiled_detect(model, frame, imgsz=None, conf=conf, classes=cls, device=dev,
                                        grid=tile_grid, overlap=overlap, region=config.TILE_REGION, full=None,
                                        upscale=config.TILE_UPSCALE)
            stats["frames"] += 1
            stats["full_small"] += int(((fb[:, 3] - fb[:, 1]) < small_h).sum()) if len(fb) else 0
            for b, c in zip(tb, tc):
                if b[3] - b[1] >= small_h or c < 0.4:
                    continue
                if len(fb) and _iou(b, fb).max() > 0.3:
                    continue
                stats["tile_only_small"] += 1
                if saved < 60:
                    x1, y1, x2, y2 = (int(v) for v in b)
                    pad = 10
                    crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                    if crop.size:
                        cv2.imwrite(str(crops_dir / f"f{idx:06d}_{x1}_{y1}_{c:.2f}.jpg"), cv2.resize(crop, None, fx=3, fy=3))
                        saved += 1
        idx += 1
    cap.release()
    tot = stats["full_small"] + stats["tile_only_small"]
    stats["est_missed_small_pct"] = round(100 * stats["tile_only_small"] / max(tot, 1), 1)
    stats["crops_dir"] = str(crops_dir)
    return stats


def _iou(b, bs):
    x1 = np.maximum(b[0], bs[:, 0]); y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2]); y2 = np.minimum(b[3], bs[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1]); ab = (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])
    return inter / np.maximum(a + ab - inter, 1e-9)


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="thư mục output của 1 video (chứa detections.csv)")
    ap.add_argument("--frame-size", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="mặc định: suy từ video (--video) hoặc 1920x1080")
    ap.add_argument("--enc-r", type=float, default=1.5, help="2 đồng đội cách nhau < N mét = gặp nhau")
    ap.add_argument("--window", type=int, default=15, help="xét đứt track / đổi đội trong ±N frame quanh lần gặp")
    ap.add_argument("--video", default=None, help="video gốc -> chạy thêm kiểm tra recall box nhỏ (F)")
    ap.add_argument("--recall-frames", type=int, default=30)
    ap.add_argument("--det-model", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--tile-grid", type=int, nargs=2, default=None, metavar=("COLS", "ROWS"))
    ap.add_argument("--small-h", type=int, default=45, help="box thấp hơn N px = cầu thủ nhỏ (F)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    run = Path(a.run_dir)
    rows, reads, meta = load(run)
    if not rows:
        raise SystemExit(f"{run}/detections.csv không có detection cầu thủ")
    W, H = a.frame_size or meta.get("frame_size") or (0, 0)
    vid = a.video or meta.get("source")
    if not W and vid and Path(vid).exists():
        import cv2
        cap = cv2.VideoCapture(str(vid))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    if not W:
        W, H = max(r["box"][2] for r in rows), max(r["box"][3] for r in rows)   # suy từ box xa nhất
        print(f"[warn] không biết kích thước frame -> ước lượng {W}x{H} từ box; truyền --frame-size cho chắc")
    fps = meta.get("fps", 25.0)
    rep = {"run": str(run), "n_det": len(rows), "n_tracks": len({r["t"] for r in rows}), "frame_size": [W, H]}

    n_det, ends, lost_events, lens, cov = size_and_breaks(rows, W, H)
    print(f"\n[{run}] {len(rows)} detection cầu thủ, {rep['n_tracks']} track, khung {W}x{H}, {fps:g} fps")
    print("\n A+B. Kích thước box và track bị đứt (đuôi track, theo chiều cao 5 box cuối)")
    print(f"   {'nhóm':>9} | {'det':>6} {'%':>5} | {'track':>5} {'dài TB':>6} | {'hết shot':>8} {'ra mép':>6} "
          f"{'đứt':>5} | {'đứt/1000 det':>12} | {'có số':>6} {'tự đọc':>6} {'nhờ nối':>7} {'k.chắc':>6}")
    rep["buckets"] = []
    for k in range(len(BUCKETS)):
        n_tr = len(lens[k])
        br = 1000 * ends["lost"][k] / max(n_det[k], 1)
        num = cov["read"][k] + cov["linked"][k] + cov["uncertain"][k]
        row = {"bucket": bname(k), "det": int(n_det[k]), "tracks": n_tr,
               "mean_len": round(float(np.mean(lens[k])), 1) if n_tr else 0,
               "end_cut": int(ends["cut"][k]), "end_edge": int(ends["edge"][k]), "end_lost": int(ends["lost"][k]),
               "lost_per_1000": round(br, 2), "pct_numbered": round(100 * num / max(n_det[k], 1), 1),
               "pct_read": round(100 * cov["read"][k] / max(n_det[k], 1), 1),
               "pct_linked": round(100 * cov["linked"][k] / max(n_det[k], 1), 1),
               "pct_uncertain": round(100 * cov["uncertain"][k] / max(n_det[k], 1), 1)}
        rep["buckets"].append(row)
        print(f"   {row['bucket']:>9} | {row['det']:>6} {100 * n_det[k] / len(rows):>4.1f}% | {n_tr:>5} "
              f"{row['mean_len']:>6} | {row['end_cut']:>8} {row['end_edge']:>6} {row['end_lost']:>5} | {br:>12.2f} | "
              f"{row['pct_numbered']:>5}% {row['pct_read']:>5}% {row['pct_linked']:>6}% {row['pct_uncertain']:>5}%")

    events = encounters(rows, a.enc_r, min_gap=3)
    oc = encounter_outcomes(events, rows, lost_events, a.window)
    n_with_xy = sum(1 for r in rows if r["xy"] is not None and r["team"] >= 0)
    rep["encounters"] = {"n": len(events), "with_lost_track": oc["lost"], "with_team_flip": oc["team_flip"],
                         "per_bucket": [sum(1 for e in events if e["bucket"] == k) for k in range(len(BUCKETS))],
                         "det_with_xy_and_team": n_with_xy}
    print(f"\n C. Gặp nhau (2 đồng đội < {a.enc_r} m, {n_with_xy} detection có toạ độ + đội): {len(events)} lần")
    if events:
        print(f"    theo kích thước: " + ", ".join(f"{bname(k)} {c}" for k, c in enumerate(rep['encounters']['per_bucket'])))
        print(f"    trong ±{a.window} frame: có track đứt giữa shot {oc['lost']} ({100 * oc['lost'] / len(events):.0f}%), "
              f"có track đổi đội {oc['team_flip']} ({100 * oc['team_flip'] / len(events):.0f}%)")
        lost_total = len(lost_events)
        lost_near = sum(1 for f, t, _ in lost_events if any(
            t in e["tracks"] and e["start"] - a.window <= f <= e["end"] + a.window for e in events))
        rep["encounters"]["lost_breaks_near_encounter_pct"] = round(100 * lost_near / max(lost_total, 1), 1)
        print(f"    {lost_near}/{lost_total} lần track đứt giữa shot xảy ra quanh một lần gặp đồng đội")

    conf_ = number_conflicts(reads, events)
    sep = [c for c in conf_ if c["separated"]]
    rep["number_conflicts"] = conf_
    print(f"\n D. Track đọc ra 2 số áo chắc chắn: {len(conf_)} (2 quãng tách biệt -> nghi hoán ID: {len(sep)}, "
          f"trong đó có gặp đồng đội ở giữa: {sum(1 for c in sep if c['encounters_between'])})")
    for c in sep[:10]:
        print(f"    track {c['track']}: #{c['numbers'][0]} frame {c['frames'][0]} / #{c['numbers'][1]} frame "
              f"{c['frames'][1]}, gặp đồng đội ở giữa: {c['encounters_between']}")

    if meta.get("link"):
        rep["link"] = meta["link"]
        ls = meta["link"]
        print(f"\n    linker: {ls['segments']} đoạn, nối {ls['links']}, bỏ mập mờ {ls['skipped_ambiguous']}, "
              f"bỏ vì đông người {ls.get('skipped_crowd', '?')}, nối qua lần gặp {ls.get('encounter_links', '-')}")

    if a.video:
        import config
        print(f"\n F. Recall box nhỏ (< {a.small_h}px): full frame vs. tile, {a.recall_frames} frame wide ...")
        rc = recall_check(a.video, rows, a.recall_frames, a.det_model or config.DET_MODEL, a.imgsz or config.DET_IMGSZ,
                          a.conf if a.conf is not None else config.DET_CONF, a.device,
                          tuple(a.tile_grid or config.TILE_GRID), config.TILE_OVERLAP, a.small_h, run)
        if rc:
            rep["recall"] = rc
            print(f"    {rc['frames']} frame: full frame thấy {rc['full_small']} cầu thủ nhỏ, tile thấy thêm "
                  f"{rc['tile_only_small']} -> ước lượng full frame bỏ sót ~{rc['est_missed_small_pct']}% "
                  f"(xem crop: {rc['crops_dir']} — có thể lẫn false positive)")

    with open(run / "far_report.json", "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n  -> {run / 'far_report.json'}")


if __name__ == "__main__":
    main()
