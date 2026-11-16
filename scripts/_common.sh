#!/usr/bin/env bash
# Dùng chung cho các script: cd về gốc repo, chọn python, parse tham số.
#   $1 = video/ảnh/folder (bắt buộc)    $2 = thư mục output (tuỳ chọn)    $3... = flag thêm cho run.py
# Biến môi trường:
#   PYTHON      python dùng để chạy (mặc định: .venv/bin/python nếu có, ngược lại python)
#   MAX_FRAMES  giới hạn số frame để test nhanh (mặc định 0 = cả video)
#   DEVICE      auto | cuda | mps | cpu (mặc định auto)
#   MODELS_DIR  thư mục chứa weights legibility + PARSeq (mặc định ./models)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi

usage() {
  echo "Usage: bash $0 <video|ảnh|folder> [output_dir] [flag thêm cho run.py ...]" >&2
  echo "  vd:  MAX_FRAMES=750 bash $0 /kaggle/input/match/match.mp4 /kaggle/working/outputs" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
SOURCE="$1"
[[ -e "$SOURCE" ]] || { echo "Không tìm thấy: $SOURCE" >&2; exit 1; }
OUT="${2:-}"
EXTRA=("${@:3}")
COMMON=(--max-frames "${MAX_FRAMES:-0}" --device "${DEVICE:-auto}")
