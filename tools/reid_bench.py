"""So sánh model re-ID cho bước nối định danh, trên output có sẵn của run.py (không chạy lại detector / tracker).

    # 1) tính embedding cho mọi box player/GK từ video gốc + detections.csv (giống run.py: mỗi --stride frame, box >= 40px)
    python tools/reid_bench.py embed outputs/<run>/<video> --video ../video1.mp4 --offset 3400 \
        --weights models/osnet_ain_x1_0_msmt17.pt models/osnet_x1_0_sportsmot_deepeiou.pt
    # 2) đo khả năng phân biệt ĐỒNG ĐỘI thuần ngoại hình trên các chỗ đứt nhân tạo của link_bench
    python tools/reid_bench.py probe outputs/<run>/<video> --emb outputs/<run>/<video>/emb_*.npz
    # 3) đo cả bước nối:  python tools/link_bench.py outputs/<run>/<video> --emb <file .npz>

probe: với mỗi chỗ đứt 'đông' (có đồng đội < 3 m quanh đầu đoạn sau B), query = embedding trung bình REID_FRAMES frame
cuối của A, gallery = B (REID_FRAMES frame đầu) + các đồng đội đó (quanh frame đầu B). Đếm:
  rank-1   = B gần A nhất;
  cho qua đúng = B gần hơn MỌI đồng đội ít nhất REID_MARGIN (đúng luật linker dùng để phá thế đám đông);
  cho qua sai  = 1 đồng đội gần hơn B và mọi đồng đội khác ít nhất REID_MARGIN (nếu đồng đội đó là ứng viên thì nối sai).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from link_bench import make_breaks  # noqa: E402
from relink import load_run  # noqa: E402


def cmd_embed(a):
    import cv2
    from pipeline.device import pick_device
    from pipeline.reid import OSNetEmbedder

    run = Path(a.run_dir)
    by_f = defaultdict(list)
    with open(run / "detections.csv") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            box = [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")]
            if fr % a.stride == 0 and r["class"] in a.classes and box[3] - box[1] >= config.REID_MIN_BOX_HEIGHT:
                by_f[fr].append((int(r["track_id"]), box))
    dev = pick_device(a.device) if a.device else pick_device()
    models = [(Path(w), OSNetEmbedder(w, dev)) for w in a.weights]
    out = {w: ([], [], []) for w, _ in models}
    cap = cv2.VideoCapture(a.video)
    for _ in range(a.offset):
        cap.grab()
    last = max(by_f)
    for fr in range(last + 1):
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"video hết ở frame {fr} (cần tới {last})")
        if fr not in by_f:
            continue
        h, w = img.shape[:2]
        crops = [img[max(b[1], 0):min(b[3], h), max(b[0], 0):min(b[2], w)] for _, b in by_f[fr]]
        for wp, m in models:
            for (t, _), e in zip(by_f[fr], m(crops)):
                out[wp][0].append(fr)
                out[wp][1].append(t)
                out[wp][2].append(e)
        if fr % 250 == 0:
            print(f"  frame {fr}/{last}", flush=True)
    for wp, (fs, ts, es) in out.items():
        p = run / f"emb_{wp.stem}.npz"
        np.savez_compressed(p, frame=np.array(fs, np.int32), track=np.array(ts, np.int32), emb=np.stack(es))
        print(f"-> {p} ({len(fs)} embedding)")


def mean_emb(emb, tid, frames):
    v = [emb[(f, tid)] for f in frames if (f, tid) in emb]
    if len(v) < 2:
        return None
    m = np.mean(np.asarray(v, np.float32), 0)
    return m / (np.linalg.norm(m) + 1e-9)


def probe(d, seeds, every, crowd_r, k, margin):
    r = defaultdict(int)
    gaps_true, gaps_rival = [], []
    for seed in range(seeds):
        dd, truth = make_breaks(d, seed, every, crowd_r=crowd_r)
        emb = dd["emb"]
        frames_of = defaultdict(list)
        for f, t in dd["dets"]:
            frames_of[t].append(f)
        for t in frames_of:
            frames_of[t].sort()
        emb_frames = defaultdict(list)
        for f, t in emb:
            emb_frames[t].append(f)
        emb_frames = {t: np.array(sorted(v)) for t, v in emb_frames.items()}
        by_f = defaultdict(list)
        for (f, t), p in dd["positions"].items():
            by_f[f].append((t, p))
        for ta, fa, tb, fb, crowded in truth:
            if not crowded:
                continue
            pb, kb = dd["positions"].get((fb, tb)), dd["team_of"].get((fb, tb), -1)
            rivals = [u for u, q in by_f[fb] if u != tb and dd["team_of"].get((fb, u)) == kb
                      and np.hypot(q[0] - pb[0], q[1] - pb[1]) < crowd_r]
            ea = mean_emb(emb, ta, frames_of[ta][-k:])
            eb = mean_emb(emb, tb, frames_of[tb][:k])
            ecs = []
            for u in rivals:
                fr = emb_frames.get(u)
                if fr is None:
                    continue
                i = int(np.searchsorted(fr, fb))
                e = mean_emb(emb, u, fr[max(0, i - k // 2): i + k // 2 + 1])
                if e is not None:
                    ecs.append(e)
            if ea is None or eb is None or not ecs:
                r["thiếu emb"] += 1
                continue
            dab = 1 - float(ea @ eb)
            dac = sorted(1 - float(ea @ e) for e in ecs)
            r["n"] += 1
            r["rank1"] += dab < dac[0]
            r["qua đúng"] += dac[0] > dab + margin
            alt = [dab] + dac[1:]                       # rival gần nhất so với B và các rival còn lại
            r["qua sai"] += min(alt) > dac[0] + margin
            gaps_true.append(dab)
            gaps_rival.append(dac[0])
    return r, np.array(gaps_true), np.array(gaps_rival)


def cmd_probe(a):
    d, _ = load_run(Path(a.run_dir))
    k, margin = config.REID_FRAMES, a.margin
    print(f"chỗ đứt 'đông' (đồng đội < {a.crowd_r} m), {a.seeds} seed, trung bình {k} frame, margin {margin}")
    for p in a.emb:
        z = np.load(p)
        d["emb"] = {(int(f), int(t)): e for f, t, e in zip(z["frame"], z["track"], z["emb"])}
        r, gt, gr = probe(d, a.seeds, a.every, a.crowd_r, k, margin)
        n = max(r["n"], 1)
        print(f"  {Path(p).stem:42s} n={r['n']:4d}  rank-1 {100 * r['rank1'] / n:5.1f}%  "
              f"cho qua đúng {100 * r['qua đúng'] / n:5.1f}%  sai {100 * r['qua sai'] / n:4.1f}%  "
              f"| d(A,B) {np.median(gt):.3f}  d(A,đồng đội gần nhất) {np.median(gr):.3f}  (thiếu emb {r['thiếu emb']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("run_dir")
    e.add_argument("--video", required=True, help="video gốc mà run_dir được chạy trên đó")
    e.add_argument("--offset", type=int, default=0, help="frame 0 của run = frame này của --video")
    e.add_argument("--weights", nargs="+", required=True)
    e.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    e.add_argument("--classes", nargs="+", default=["player", "goalkeeper"])
    e.add_argument("--device", default=None)
    p = sub.add_parser("probe")
    p.add_argument("run_dir")
    p.add_argument("--emb", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--crowd-r", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=config.REID_MARGIN)
    a = ap.parse_args()
    cmd_embed(a) if a.cmd == "embed" else cmd_probe(a)


if __name__ == "__main__":
    main()
