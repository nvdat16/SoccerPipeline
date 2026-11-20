"""Chạy lại bước định danh (IdentityLinker) từ output CSV của run.py — không cần GPU, không chạy lại video.

    python tools/relink.py outputs/<run>/<video>                       # -> outputs/<run>/<video>/relink/
    python tools/relink.py outputs/<run>/<video> --no-enc-split --out /tmp/x
    python tools/relink.py outputs/<run>/<video> --set ENC_R=2.0 LINK_AMBIGUITY=0.2

Đọc detections.csv (đội, class), positions.csv (vị trí đã làm mượt, role sau khi chia đội), reads.csv, tracks.json
(fps). Ghi detections.csv / positions.csv / tracks.json mới (cột jersey_number, number_source, player_id thay bằng
kết quả nối mới) + links.json, để chạy tools/far_report.py trên đó và so sánh tham số.
Lưu ý: số áo của trọng tài / detection không thuộc đoạn nào giữ nguyên như file gốc.
"""
import argparse
import csv
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline.identity import IdentityLinker  # noqa: E402


def role_of(name):
    n = str(name).lower()
    if n in ("goalkeeper", "goalie"):
        return "goalkeeper"
    if n in config.REFEREE_CLASS_NAMES:
        return "referee"
    return "player"


def load_run(run: Path):
    """Đọc output của run.py -> (dict dữ liệu cho IdentityLinker.run + các dòng CSV gốc, fps)."""
    meta = json.loads((run / "tracks.json").read_text())
    with open(run / "detections.csv") as f:
        det_rows = list(csv.DictReader(f))
    with open(run / "positions.csv") as f:
        pos_rows = list(csv.DictReader(f))
    with open(run / "reads.csv") as f:
        reads = [(int(r["frame"]), int(r["track_id"]), r["label"], float(r["prob"])) for r in csv.DictReader(f)]
    dets = [(int(r["frame"]), int(r["track_id"])) for r in det_rows]
    team_of = {(int(r["frame"]), int(r["track_id"])): int(r["team"]) for r in det_rows}
    positions = {(int(r["frame"]), int(r["track_id"])): (float(r["x_m"]), float(r["y_m"])) for r in pos_rows}
    role_pos = {int(r["track_id"]): r["role"] for r in pos_rows}           # sau khi chia đội (GK -> player)
    cls_votes = defaultdict(Counter)
    for r in det_rows:
        cls_votes[int(r["track_id"])][r["class"]] += 1
    roles = {t: role_pos.get(t) or role_of(v.most_common(1)[0][0]) for t, v in cls_votes.items()}
    return {"meta": meta, "det_rows": det_rows, "pos_rows": pos_rows, "reads": reads, "dets": dets,
            "team_of": team_of, "positions": positions, "roles": roles}, meta.get("fps", 25.0)


def make_cfg(overrides=(), enc_split=None, sum_th=None):
    """Hằng số config.py (+ override KEY=VALUE) -> (cfg, voter_kw)."""
    cfg = types.SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
    if enc_split is not None:
        cfg.ENC_SPLIT = enc_split
    for kv in overrides:
        k, v = kv.split("=", 1)
        if not hasattr(cfg, k):
            raise SystemExit(f"--set: config không có {k}")
        old = getattr(cfg, k)
        setattr(cfg, k, v.lower() in ("1", "true") if isinstance(old, bool) else type(old)(v))
    voter_kw = dict(filter_threshold=cfg.FILTER_THRESHOLD, sum_threshold=sum_th or cfg.SUM_THRESHOLD,
                    use_bias=cfg.USE_BIAS)
    return cfg, voter_kw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="mặc định <run_dir>/relink")
    ap.add_argument("--enc-split", action=argparse.BooleanOptionalAction, default=config.ENC_SPLIT)
    ap.add_argument("--sum-th", type=float, default=config.SUM_THRESHOLD)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override hằng số trong config.py")
    a = ap.parse_args()

    run = Path(a.run_dir)
    out = Path(a.out) if a.out else run / "relink"
    out.mkdir(parents=True, exist_ok=True)
    d, fps = load_run(run)
    det_rows, pos_rows, reads = d["det_rows"], d["pos_rows"], d["reads"]
    det_header = list(det_rows[0].keys()) if det_rows else []
    pos_header = list(pos_rows[0].keys()) if pos_rows else []
    meta = d["meta"]
    cfg, voter_kw = make_cfg(a.set, enc_split=a.enc_split, sum_th=a.sum_th)
    linker = IdentityLinker(fps, cfg, voter_kw)
    ident, players, stats = linker.run(d["dets"], d["positions"], d["team_of"], d["roles"], reads)

    def patch(rows, header, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                k = (int(r["frame"]), int(r["track_id"]))
                if k in ident:
                    pid, num, src = ident[k]
                    r = {**r, "player_id": pid, "jersey_number": num, "number_source": src}
                w.writerow(r)

    patch(det_rows, det_header, out / "detections.csv")
    patch(pos_rows, pos_header, out / "positions.csv")
    with open(out / "reads.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "label", "prob"])
        w.writerows(reads)
    meta = {**meta, "link": stats, "players": players, "relink": {"enc_split": a.enc_split, "set": a.set}}
    (out / "tracks.json").write_text(json.dumps(meta, indent=2))
    (out / "links.json").write_text(json.dumps(linker.links, indent=1))

    n = max(stats["dets_total"], 1)
    print(f"[{run}] enc_split={a.enc_split} {' '.join(a.set)}")
    print(f"  {stats['segments']} đoạn -> nối {stats['links']} ({stats['solver']}; qua lần gặp {stats['encounter_links']}), bỏ mập mờ "
          f"{stats['skipped_ambiguous']}, bỏ đông người {stats['skipped_crowd']}, khác số {stats['skipped_number_conflict']}")
    print(f"  gặp đồng đội {stats['encounters']} lần, đoạn E gắn lại {stats['encounter_segments_attached']}/"
          f"{stats['encounter_segments']}")
    print(f"  detection có số: tự đọc {100 * stats['dets_read'] / n:.1f}% + nhờ nối {100 * stats['dets_linked'] / n:.1f}%"
          f" + không chắc {100 * stats['dets_uncertain'] / n:.1f}% | {stats['players_with_number']} cầu thủ có số | "
          f"trùng ID trong 1 frame: {stats['dup_frame_ids']} | hạ số (chồng thời gian): {stats['demoted_same_frame']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
