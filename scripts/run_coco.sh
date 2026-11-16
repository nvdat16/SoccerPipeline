#!/usr/bin/env bash
# YOLO COCO pose (yolo11m-pose) vừa detect người vừa cho keypoints -> crop thân áo theo vai/hông.
# Bắt được cầu thủ ở cảnh cận (yolo26s-soccer hay bỏ sót) nhưng cũng bắt cả khán giả, trọng tài.
#   bash scripts/run_coco.sh <video> [output_dir] [flag thêm...]
#   COCO_MODEL=yolo26m-pose.pt bash scripts/run_coco.sh <video>
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" run.py --source "$SOURCE" --output "${OUT:-outputs/coco}" "${COMMON[@]}" \
  --det-model "${COCO_MODEL:-yolo11m-pose.pt}" --torso-mode pose \
  ${EXTRA[@]+"${EXTRA[@]}"}
