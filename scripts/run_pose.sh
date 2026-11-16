#!/usr/bin/env bash
# yolo26s-soccer -> ByteTrack -> legibility -> yolo26n-pose lấy vai/hông -> crop thân áo -> PARSeq -> vote.
# Cách cắt giống paper gốc; chậm hơn bbox và bỏ qua cầu thủ có keypoint yếu.
#   bash scripts/run_pose.sh <video> [output_dir] [flag thêm...]
#   POSE_MODEL=yolo26s-pose.pt bash scripts/run_pose.sh <video>    # pose model lớn hơn
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" run.py --source "$SOURCE" --output "${OUT:-outputs/pose}" "${COMMON[@]}" \
  --det-model yolo26s-soccer.pt --torso-mode pose --pose-model "${POSE_MODEL:-yolo26n-pose.pt}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
