#!/usr/bin/env bash
# [KHUYẾN NGHỊ] yolo26s-soccer -> ByteTrack -> legibility -> crop thân áo theo tỉ lệ bbox -> PARSeq -> vote.
# Không dùng model pose. Trên liver-mu.mp4: nhận ra 35 track (pose: 22), nhanh hơn ~5%.
#   bash scripts/run_bbox.sh <video> [output_dir] [flag thêm...]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" run.py --source "$SOURCE" --output "${OUT:-outputs/bbox}" "${COMMON[@]}" \
  --det-model yolo26s-soccer.pt --torso-mode bbox \
  ${EXTRA[@]+"${EXTRA[@]}"}
