"""Cấu hình mặc định cho pipeline. Mọi giá trị đều có thể override qua CLI (xem run.py)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Trên Kaggle có thể trỏ tới Kaggle Dataset chứa weights: MODELS_DIR=/kaggle/input/<dataset>
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

# Code PARSeq (strhub) được vendor sẵn trong repo
PARSEQ_DIR = ROOT / "third_party" / "parseq"

# ---------------------------------------------------------------- models
# Detector + tracker. Nếu là model *-pose thì keypoints có sẵn luôn, khỏi chạy pose riêng.
# Có thể thay bằng model detect cầu thủ tự train (khi đó set --classes cho đúng).
DET_MODEL = "yolo11m-pose.pt"
POSE_MODEL = "yolo11m-pose.pt"          # chỉ dùng khi DET_MODEL không phải model pose
PLAYER_CLASSES = [0]                    # COCO: 0 = person
TRACKER = "bytetrack.yaml"

LEGIBILITY_FILE = "legibility_resnet34_soccer_20240215.pth"
STR_FILE = "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"

# Weights của paper (Koshkina & Elder, CVPRW 2024) — fine-tune trên SoccerNet
MODEL_URLS = {
    LEGIBILITY_FILE: "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
    STR_FILE: "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
}

# ---------------------------------------------------------------- detection
DET_CONF = 0.3
DET_IMGSZ = 1280                        # cầu thủ trong video broadcast khá nhỏ -> imgsz lớn

# ---------------------------------------------------------------- jersey recognition
FRAME_STRIDE = 2                        # chỉ đọc số áo mỗi N frame (tracking vẫn chạy mọi frame)
MIN_BOX_HEIGHT = 60                     # bỏ cầu thủ quá nhỏ (px)
LEGIBILITY_THRESHOLD = 0.5
KPT_CONF = 0.4                          # ngưỡng confidence keypoint vai/hông (như repo gốc)
TORSO_PADDING = 5
MAX_READS_PER_TRACK = 60                # đủ mẫu thì ngừng đọc track đó để tiết kiệm thời gian

# Gộp kết quả theo tracklet (giống helpers.find_best_prediction của repo gốc)
FILTER_THRESHOLD = 0.2                  # bỏ dự đoán có xác suất < 0.2
SUM_THRESHOLD = 1.0                     # tổng trọng số tối thiểu để chốt số áo
USE_BIAS = True                         # ưu tiên số 2 chữ số (0.61 vs 0.39)
