"""Benchmark bước nối định danh CÓ ĐÁP ÁN, từ output của run.py (không cần GPU, không chạy lại video).

    python tools/link_bench.py outputs/<run>/<video>
    python tools/link_bench.py outputs/<run>/<video> --set LINK_SOLVER=greedy LINK_AMBIGUITY=0.15

Không có nhãn thật về danh tính, nên tự tạo: cắt nhân tạo các track dài (mỗi ~every frame bỏ đi g frame, phần
sau mang track id mới) -> biết chắc đoạn trước và đoạn sau là 1 người. Chạy IdentityLinker trên dữ liệu đã cắt,
ẨN số áo (chỉ đo phần chuyển động) và đếm:
  nối đúng = cuối đoạn trước được nối đúng vào đầu đoạn sau;
  nối sai  = cuối đoạn trước nối vào đoạn khác, hoặc đầu đoạn sau nhận đoạn khác.
Tách ca 'đông' (có đồng đội < 3 m ở cuối khoảng trống) / 'thoáng'. Lặp nhiều seed (vị trí cắt, độ dài khoảng trống).
Lưu ý: chỗ đứt thật hay xảy ra lúc bị che khuất (khó hơn) -> con số ở đây là so sánh TƯƠNG ĐỐI giữa các cấu hình.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relink import load_run, make_cfg  # noqa: E402

from pipeline.identity import IdentityLinker  # noqa: E402


def make_breaks(d, seed=0, every=50, gaps=(8, 15, 25, 40), crowd_r=3.0):
    """-> (dữ liệu đã cắt, đáp án [(track A, frame cuối A, track B, frame đầu B, đông?)])."""
    rng = np.random.default_rng(seed)
    pos, team = d["positions"], d["team_of"]
    by_t = defaultdict(list)
    for f, t in d["dets"]:
        by_t[t].append(f)
    by_f = defaultdict(list)
    for (f, t), p in pos.items():
        by_f[f].append((t, p))
    newid, truth, drop = {}, [], set()
    nxt = max(by_t) + 100000
    for t, fs in by_t.items():
        if d["roles"].get(t) == "referee" or len(fs) < 2 * every:
            continue
        fs = sorted(fs)
        have = set(fs)
        cuts, i = [], every
        while i < len(fs) - every // 2:
            g, c = int(rng.choice(gaps)), fs[i]
            if all(f in have for f in range(c - 10, c + g + 10)):
                cuts.append((c, g))
                i += every
            else:
                i += 10
        cur, last = t, 0
        for c, g in cuts:
            for f in fs:
                if last <= f < c:
                    newid[(f, t)] = cur
            drop.update((f, t) for f in range(c, c + g))
            fb, p, k = c + g, pos.get((c + g, t)), team.get((c + g, t), -1)
            crowded = p is not None and any(u != t and team.get((fb, u)) == k and np.hypot(q[0] - p[0], q[1] - p[1]) < crowd_r
                                            for u, q in by_f[fb])
            truth.append((cur, c - 1, nxt, fb, crowded))
            last, cur, nxt = fb, nxt, nxt + 1
        for f in fs:
            if f >= last and (f, t) not in drop:
                newid[(f, t)] = cur

    def m(k):
        return k[0], newid.get(k, k[1])

    dd = dict(d)
    dd["dets"] = [m(k) for k in d["dets"] if k not in drop]
    dd["positions"] = {m(k): v for k, v in pos.items() if k not in drop}
    dd["team_of"] = {m(k): v for k, v in team.items() if k not in drop}
    roles = dict(d["roles"])
    for (f, t), v in newid.items():
        roles.setdefault(v, d["roles"].get(t, "player"))
    dd["roles"] = roles
    if d.get("emb"):
        dd["emb"] = {m(k): v for k, v in d["emb"].items() if k not in drop}
    return dd, truth


def score(dd, truth, cfg, voter_kw, fps):
    L = IdentityLinker(fps, cfg, voter_kw)
    L.run(dd["dets"], dd["positions"], dd["team_of"], dd["roles"], [], emb=dd.get("emb"))   # ẩn số áo
    out = {(l["from_track"], l["from_frame"]): (l["to_track"], l["to_frame"]) for l in L.links}
    into = {v: k for k, v in out.items()}
    r = defaultdict(int)
    r["reid", "resolved"] = L.n_reid
    for ta, fa, tb, fb, crowded in truth:
        g = "đông" if crowded else "thoáng"
        r[g, "n"] += 1
        got = out.get((ta, fa))
        if got == (tb, fb):
            r[g, "ok"] += 1
            continue
        r[g, "bad"] += got is not None
        src = into.get((tb, fb))
        r[g, "bad"] += src is not None and src != (ta, fa)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--every", type=int, default=50, help="cắt mỗi N frame của track")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override hằng số trong config.py")
    ap.add_argument("--emb", default=None, help="file .npz embedding re-ID (tools/reid_bench.py embed) -> linker dùng re-ID")
    a = ap.parse_args()

    d, fps = load_run(Path(a.run_dir))
    if a.emb:
        z = np.load(a.emb)
        d["emb"] = {(int(f), int(t)): e for f, t, e in zip(z["frame"], z["track"], z["emb"])}
    cfg, voter_kw = make_cfg(a.set)
    tot = defaultdict(int)
    for seed in range(a.seeds):
        dd, truth = make_breaks(d, seed, a.every)
        for k, v in score(dd, truth, cfg, voter_kw, fps).items():
            tot[k] += v
    print(f"solver={cfg.LINK_SOLVER} accept={cfg.LINK_ACCEPT} ambiguity={cfg.LINK_AMBIGUITY} "
          f"crowd_margin={cfg.LINK_CROWD_MARGIN} ({a.seeds} seed)")
    ok = bad = n = 0
    for g in ("thoáng", "đông"):
        n_, ok_, bad_ = tot[g, "n"], tot[g, "ok"], tot[g, "bad"]
        n, ok, bad = n + n_, ok + ok_, bad + bad_
        print(f"  {g:7s}: {n_:4d} chỗ đứt, nối lại đúng {ok_:4d} ({100 * ok_ / max(n_, 1):4.1f}%), nối sai {bad_}")
    print(f"  tổng   : {n:4d} chỗ đứt, nối lại đúng {ok:4d} ({100 * ok / max(n, 1):4.1f}%), nối sai {bad}, "
          f"độ chính xác {100 * ok / max(ok + bad, 1):.1f}%")
    if a.emb:
        print(f"  re-ID  : {tot['reid', 'resolved']} cặp đám đông được ngoại hình cho qua ({Path(a.emb).name})")


if __name__ == "__main__":
    main()
