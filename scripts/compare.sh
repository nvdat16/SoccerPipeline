#!/usr/bin/env bash
# Chạy các luồng (bbox, pose, no_legibility, coco) trên cùng 1 video rồi in bảng so sánh.
#   MAX_FRAMES=2500 bash scripts/compare.sh <video> [output_dir]
#   CASES="bbox pose" bash scripts/compare.sh <video>          # chỉ chạy một số luồng
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT="${OUT:-outputs/compare}"
CASES="${CASES:-bbox pose no_legibility coco}"
DIR="$(dirname "${BASH_SOURCE[0]}")"
for c in $CASES; do
  echo "================ $c"
  bash "$DIR/run_$c.sh" "$SOURCE" "$OUT/$c" --no-video ${EXTRA[@]+"${EXTRA[@]}"}
done

"$PYTHON" - "$OUT" "$SOURCE" $CASES <<'PY'
import json, sys, collections
from pathlib import Path
out, src, cases = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
stem = src.stem if src.is_file() else None
print(f"\n{'luồng':<15}{'tracks':>8}{'có số':>8}{'#10':>6}{'str_reads':>11}{'no_pose':>9}{'fps':>8}   số áo (số track)")
for c in cases:
    d = out / c / stem if stem else next((out / c).iterdir())
    t = json.load(open(d / "tracks.json"))["tracks"]
    tm = json.load(open(d / "timing.json"))
    nums = [v["jersey_number"] for v in t.values() if v["jersey_number"] > 0]
    cnt = dict(sorted(collections.Counter(nums).items()))
    print(f"{c:<15}{len(t):>8}{len(nums):>8}{cnt.get(10, 0):>6}{sum(v['str_reads'] for v in t.values()):>11}"
          f"{sum(v['no_pose'] for v in t.values()):>9}{tm['analyse_fps']:>8}   {cnt}")
print("\n'#10' cao bất thường = PARSeq đang bịa số (thường do bỏ legibility hoặc crop lẫn logo/tay áo).")
PY
