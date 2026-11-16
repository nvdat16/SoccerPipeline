#!/usr/bin/env bash
# Cài dependencies + tải weights legibility/PARSeq. Chạy 1 lần sau khi clone (Kaggle hoặc local).
#   bash scripts/setup.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m pip install -q -r requirements.txt
"$PYTHON" download_models.py
"$PYTHON" -c "import torch, ultralytics; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)"
