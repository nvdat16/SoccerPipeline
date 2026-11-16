#!/usr/bin/env bash
# [DEBUG] bỏ legibility, đưa mọi crop thân áo (theo bbox) thẳng vào PARSeq.
# KHÔNG dùng cho kết quả thật: trên liver-mu.mp4 ra 35 track "#10" và 17 track "#27" (PARSeq bịa số).
# Dùng để xem PARSeq đọc được gì khi legibility loại quá tay.
#   bash scripts/run_no_legibility.sh <video> [output_dir] [flag thêm...]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" run.py --source "$SOURCE" --output "${OUT:-outputs/no_legibility}" "${COMMON[@]}" \
  --det-model yolo26s-soccer.pt --torso-mode bbox --no-legibility --save-crops \
  ${EXTRA[@]+"${EXTRA[@]}"}
