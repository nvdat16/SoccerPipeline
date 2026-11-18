#!/usr/bin/env bash
# Luồng bbox (khuyến nghị) + MINIMAP 2D: phát hiện cắt cảnh -> pitch keypoints (32 điểm mốc) -> homography
# ảnh->sân -> chiếu chân cầu thủ lên sân (mét) -> minimap có số áo ở góc video + positions.csv.
#   bash scripts/run_minimap.sh <video> [output_dir] [flag thêm...]
#   MAX_FRAMES=1500 bash scripts/run_minimap.sh match.mp4 outputs/minimap --pitch-debug   # kiểm tra homography
#   PITCH_SIZE="101 68" bash scripts/run_minimap.sh match.mp4                            # kích thước sân thật
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
read -r -a SIZE <<< "${PITCH_SIZE:-105 68}"
"$PYTHON" run.py --source "$SOURCE" --output "${OUT:-outputs/minimap}" "${COMMON[@]}" \
  --det-model yolo26s-soccer.pt --torso-mode bbox \
  --minimap --pitch-size "${SIZE[@]}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
