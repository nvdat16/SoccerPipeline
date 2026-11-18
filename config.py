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
# Có thể thay bằng model detect cầu thủ tự train (vd yolo26s-soccer.pt).
# Mặc định: detector bóng đá tự train (có trong repo); không có thì dùng YOLO COCO pose.
_SOCCER_DET = ROOT / "yolo26s-soccer.pt"    # classes: 0 ball, 1 goalkeeper, 2 player, 3 referee
DET_MODEL = str(_SOCCER_DET) if _SOCCER_DET.exists() else "yolo11m-pose.pt"
POSE_MODEL = "yolo26n-pose.pt"          # chỉ dùng khi --torso-mode pose và DET_MODEL không phải model pose
# Class cầu thủ được chọn THEO TÊN trong model.names (không dùng id cứng: model tự train
# thường có 0 = ball). COCO -> 'person'; model bóng đá -> 'player', 'goalkeeper'.
# Muốn đọc số trọng tài thì thêm 'referee' hoặc truyền --classes player goalkeeper referee.
PLAYER_CLASS_NAMES = ["person", "player", "goalkeeper", "goalie"]
TRACKER = "bytetrack.yaml"

LEGIBILITY_FILE = "legibility_resnet34_soccer_20240215.pth"
STR_FILE = "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"
# Minimap: calibration camera <-> sân
#   pnlcalib: PnLCalib (HRNet keypoints 57 + vạch có tên 23, train SoccerNet) — mặc định
#   roboflow: YOLOv8x-pose 32 điểm (nhanh hơn nhưng nhầm vòng cấm thành giữa sân khi camera zoom gần)
PNL_KP_FILE = "pnlcalib_SV_kp.pth"
PNL_LINE_FILE = "pnlcalib_SV_lines.pth"
PITCH_FILE = "football-pitch-detection.pt"

# Weights của paper (Koshkina & Elder, CVPRW 2024) — fine-tune trên SoccerNet
# + pitch keypoints của Roboflow (https://github.com/roboflow/sports, examples/soccer/setup.sh)
MODEL_URLS = {
    LEGIBILITY_FILE: "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
    STR_FILE: "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
    PITCH_FILE: "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
    # PnLCalib v1.0.0 (https://github.com/mguti97/PnLCalib), weights CC BY 4.0, ~265 MB mỗi file
    PNL_KP_FILE: "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp",
    PNL_LINE_FILE: "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines",
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
# --torso-mode: 'pose' = cắt theo keypoint vai/hông (như paper), 'bbox' = cắt theo tỉ lệ bbox, không cần pose.
# So sánh trên liver-mu.mp4 (2500 frame): bbox nhận ra 35 track vs pose 22, nhanh hơn ~5%, 0 crop mất vì thiếu keypoint.
TORSO_MODE = "bbox"
# Tỉ lệ (x1, y1, x2, y2) của thân áo trong bbox. Đo trên 636 cầu thủ (liver-mu.mp4, yolo26s-soccer):
# vai ~0.25, hông ~0.50 chiều cao; thân 0.34-0.66 chiều rộng (p10-p90: 0.24-0.76) -> nới rộng một chút.
TORSO_BBOX_RATIO = (0.22, 0.20, 0.78, 0.52)
MAX_READS_PER_TRACK = 60                # đủ mẫu thì ngừng đọc track đó để tiết kiệm thời gian

# Gộp kết quả theo tracklet (giống helpers.find_best_prediction của repo gốc)
FILTER_THRESHOLD = 0.2                  # bỏ dự đoán có xác suất < 0.2
SUM_THRESHOLD = 1.0                     # tổng trọng số tối thiểu để chốt số áo
USE_BIAS = True                         # ưu tiên số 2 chữ số (0.61 vs 0.39)

# ---------------------------------------------------------------- minimap (--minimap)
# Video broadcast = 1 feed nhiều góc máy -> tính lại phép chiếu ảnh->sân mỗi frame, reset ở mỗi cắt cảnh.
CALIB_BACKEND = "pnlcalib"              # pnlcalib | roboflow
PNL_REFINE = False                      # PnLCalib: tinh chỉnh homography theo vạch (chậm hơn)
PNL_MAX_REPROJ_PX = 8.0                 # PnLCalib: lỗi reprojection RMS tối đa (px, frame gốc)
PITCH_SIZE = (105.0, 68.0)              # dài x rộng (m). Anfield 101x68, Old Trafford 105x68...
PITCH_IMGSZ = 640                       # model pitch được train ở 640
PITCH_DET_CONF = 0.3
PITCH_KPT_CONF = 0.5                    # chỉ dùng điểm mốc có conf >= ngưỡng này
PITCH_EVERY = {"pnlcalib": 5,           # chạy calibration mỗi N frame; giữa các lần nối bằng optical flow
               "roboflow": 1}           # (PnLCalib: 2 HRNet-w48 ~0.8 s/frame trên M1, nhanh hơn nhiều trên T4)
PITCH_RANSAC_M = 1.0                    # ngưỡng inlier RANSAC (mét trên sân)
PITCH_MAX_REPROJ_M = 1.0                # lỗi reprojection trung vị tối đa để tin homography
PITCH_MAX_JUMP_M = 5.0                  # H mới lệch H dự đoán > N mét -> coi là sai (trong cùng shot)
PITCH_SWITCH_AFTER = 12                 # ...trừ khi các lần đo trong N frame liên tiếp cùng chỉ về vị trí mới
PITCH_EMA = 0.5                         # trọng số H đo được khi trộn với H dự đoán từ optical flow
PITCH_MAX_FLOW_FRAMES = 50              # nối bằng optical flow tối đa N frame (~2 s) khi thiếu điểm mốc
PITCH_OUTSIDE_M = 3.0                   # cầu thủ chiếu ra ngoài sân quá N mét -> bỏ (ghế dự bị, khán đài)
MINIMAP_WIDTH = 420                     # px
MINIMAP_POS = "bottom-right"            # bottom-left | bottom-center | bottom-right | top-left | ...
MINIMAP_STALE_FRAMES = 75               # frame không có phép chiếu: giữ chấm mờ tối đa N frame (~3 s)
REFEREE_CLASS_NAMES = ["referee"]       # --minimap: track thêm trọng tài (vẽ màu đen, không đọc số)
