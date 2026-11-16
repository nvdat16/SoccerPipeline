#!/usr/bin/env bash
# Chẩn đoán vì sao số áo không được nhận diện: chạy mọi bước trên mọi cầu thủ (không lọc),
# ghi attempts.csv + lưu crop "PARSeq đọc chắc chắn mà vẫn bị chặn".
#   bash scripts/diagnose.sh <video> [output_dir] [flag thêm cho tools/diagnose.py...]
#   EVERY=5 START=600 MAX_FRAMES=200 bash scripts/diagnose.sh <video>
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" tools/diagnose.py --source "$SOURCE" --out "${OUT:-outputs/diagnose}" \
  --det-model yolo26s-soccer.pt --pose-model "${POSE_MODEL:-yolo26n-pose.pt}" \
  --every "${EVERY:-10}" --start "${START:-0}" --max-frames "${MAX_FRAMES:-0}" --device "${DEVICE:-auto}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
