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
# ByteTrack với ngưỡng chỉnh cho bóng đá (trackers/bytetrack_soccer.yaml). Đo trên clip_3400 (1500 frame):
#   bytetrack.yaml + conf 0.3 (cũ)  : 10.5 lần đứt track giữa shot / 1000 detection, số áo phủ 43.9% detection
#   bytetrack_soccer.yaml + conf 0.1: 7.5 / 1000 (-29%), số áo phủ 45.8%                       <- mặc định
#   botsort_soccer.yaml (thêm GMC)  : 7.5 / 1000, số áo phủ 38.1%, tracker chậm gấp ~3 lần
TRACKER = str(ROOT / "trackers" / "bytetrack_soccer.yaml")

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
    # OSNet-AIN x1.0 train MSMT17 (model zoo của deep-person-reid / torchreid, MIT) — chỉ cần với --reid
    "osnet_ain_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
    # OSNet x1.0 của Deep-EIoU (Huang et al., WACVW 2024), fine-tune trên crop cầu thủ SportsMOT (sports_model.pth.tar-60;
    # repo không ghi license, SportsMOT chỉ cho nghiên cứu) — REID_FILE mặc định của --reid
    "osnet_x1_0_sportsmot_deepeiou.pt": "https://drive.google.com/uc?id=14zzlm1nI9Ws_Il9RYNChwPC7Fsul7xwl",
}

# ---------------------------------------------------------------- detection
# Ngưỡng của DETECTOR (lọc trước tracker). Để thấp: tracker tự chia ngưỡng (track_high/low/new_track_thresh
# trong file yaml). Với 0.3 (cũ) > track_low_thresh, bước ghép thứ 2 của BYTE không bao giờ nhận được box nào.
DET_CONF = 0.1
DET_IMGSZ = 1280                        # cầu thủ trong video broadcast khá nhỏ -> imgsz lớn
# --tile: detect thêm trên tile (vùng xa camera) để bắt cầu thủ nhỏ. Chậm hơn ~(1 + số tile) lần ở bước detect.
# Chỉ bật khi tools/far_report.py --video cho thấy full frame bỏ sót nhiều cầu thủ nhỏ.
TILE = False
TILE_GRID = (3, 1)                      # (cột, hàng)
TILE_REGION = (0.0, 0.0, 1.0, 0.6)      # vùng cắt tile (tỉ lệ x1, y1, x2, y2): nửa trên = phía xa camera chính
TILE_OVERLAP = 0.2
TILE_UPSCALE = 1.5                      # tile được phóng to N lần so với ảnh gốc trước khi detect
TILE_MAX_H = 60                         # chỉ lấy box từ tile thấp hơn N px (box lớn thì full frame đã thấy)
TILE_IOS = 0.5                          # box tile trùng box đã có (giao / box nhỏ hơn >= N) -> bỏ

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

# ---------------------------------------------------------------- chia đội (--teams)
# Histogram HSV vùng thân áo -> KMeans 2 cụm trên mọi track 'player' -> bỏ phiếu theo track.
TEAM_BBOX_RATIO = (0.25, 0.15, 0.75, 0.50)  # vùng thân áo lấy màu (tỉ lệ bbox), hẹp để ít cỏ
TEAM_MIN_BOX_HEIGHT = 30                # cầu thủ nhỏ hơn N px không lấy mẫu màu
TEAM_FIT_SAMPLES = 30                   # KMeans dùng tối đa N mẫu / track (mỗi --stride frame lấy 1 mẫu)
TEAM_MIN_SAMPLES = 3                    # track ít mẫu hơn không dùng để fit (vẫn được gán đội)
TEAM_OUTLIER_MAD = 4.0                  # mẫu xa tâm đội > median + N*MAD -> màu lạ, team = -1
TEAM_SMOOTH = 4                         # làm mượt nhãn đội: đa số trong ±N mẫu của cùng track

# ---------------------------------------------------------------- định danh cầu thủ trên minimap (--link)
# Nối các đoạn track thành 1 cầu thủ bằng vị trí + vận tốc + hướng trên sân (mét); số áo đọc rõ ở 1 đoạn
# được truyền sang cả chuỗi. Cần --minimap.
LINK_MAX_GAP_S = 3.0                    # nối 2 đoạn cách nhau tối đa N giây
LINK_PREDICT_S = 1.0                    # ngoại suy theo vận tốc tối đa N giây, sau đó coi như đứng yên
LINK_R0 = 2.0                           # bán kính cho phép (m) khi khoảng trống = 0 (nhiễu phép chiếu)
LINK_SPREAD = 3.0                       # bán kính nở thêm N m mỗi giây mất dấu
LINK_R_MAX = 10.0                       # bán kính tối đa (m)
LINK_W_DIR = 0.5                        # trọng số độ lệch hướng chạy (cả 2 đoạn đang chạy)
LINK_MOVING_MS = 1.5                    # coi là "đang chạy" khi tốc độ > N m/s
LINK_VEL_POINTS = 10                    # số điểm cuối/đầu đoạn để ước lượng vận tốc
LINK_VMAX = 10.0                        # giới hạn tốc độ ước lượng (m/s), chặn nhiễu phép chiếu
LINK_W_VEL = 0.3                        # trọng số đổi vận tốc đột ngột
# Ghép: 'global' = Hungarian trên từng cụm ứng viên (max Σ LINK_ACCEPT - chi phí), 'greedy' = tham lam (bản cũ).
# Benchmark cắt track nhân tạo trên clip_3400 (518 chỗ đứt có đáp án, chỉ dùng chuyển động, tools/link_bench.py):
#                                                 benchmark                       | dữ liệu thật: số áo nhờ nối
#   greedy, ambiguity 0.15, crowd 1.0 (cũ)       : đúng 56%, sai 48 (chính xác 85.8%) | 5.3% detection
#   global, accept 1.2, ambiguity 0.3, crowd 0.5 : đúng 72%, sai 37 (91.0%)           | 5.2%   <- mặc định
#   global, accept 1.0, ambiguity 0.3, crowd 0.5 : đúng 70%, sai 26 (93.3%)           | 4.1%
#   global, accept 1.2, ambiguity 0.15, crowd 0.5: đúng 77%, sai 47 (89.5%)           | 5.6%
LINK_SOLVER = "global"
LINK_ACCEPT = 1.2                       # chỉ nối cặp có chi phí < N (khoảng cách dự đoán = bán kính cho phép -> 1)
LINK_AMBIGUITY = 0.3                    # bỏ cặp mà phương án tốt nhất không có nó chỉ kém < N -> mập mờ, không nối
LINK_CROWD_MARGIN = 0.5                 # cầu thủ cùng đội khác cách vị trí dự đoán <= (khoảng cách tới B + N m) -> không nối

# ---------------------------------------------------------------- re-ID ngoại hình (--reid, mặc định tắt)
# OSNet-AIN (MSMT17) chỉ dùng để phân định cặp bị luật đám đông chặn: ngoại hình cuối A giống B hơn mọi đồng đội
# đứng gần > REID_MARGIN (khoảng cách cosine) -> cho nối, số áo truyền qua đó là 'uncertain'.
# Đo trên clip_3400: trong đám đông ngoại hình chỉ chọn đúng ~62% (đồng đội mặc áo giống hệt; LDA theo video không
# giúp), benchmark cắt track: +5 nối đúng / +3 nối sai -> mặc định tắt. Giữ lại để thay model re-ID riêng cho bóng đá.
# 2026-11-25, Deep-EIoU OSNet (SportsMOT) trên clip_3400 (tools/reid_bench.py + link_bench --emb):
#   phân biệt đồng đội < 3 m: rank-1 67.9% (MSMT17) -> 89.3%; link_bench tổng 387 đúng/33 sai (không re-ID)
#   -> 392/22 (MSMT17: 380/32). Nhưng chỗ đứt THẬT: chỉ 5/96 cặp đám đông được phân định (45 cặp B quá ngắn, thiếu emb)
#   -> vẫn tắt mặc định cho tới khi kiểm tra bằng mắt.
#   SoccerNet-ReID OSNet (models/osnet_x1_0_soccernet_reid.pt, sn-reid baseline): rank-1 80.4%, link_bench 390/26;
#   ở margin >= 0.1 không cho qua cặp sai nào. Chi tiết: experiments/reid_2026-11-25/RESULTS.md
REID = False
REID_FILE = "osnet_x1_0_sportsmot_deepeiou.pt"
REID_MARGIN = 0.06
REID_FRAMES = 10                        # trung bình embedding N frame cuối A / đầu B
REID_MIN_BOX_HEIGHT = 40                # px
# Gặp nhau: 2 đồng đội cách nhau < ENC_R m -> tracker hay hoán ID / làm đứt track. Cắt track ở đó (A | E | B),
# rồi quyết định lại A -> B bằng chuyển động trước lần gặp (tin tracker trừ khi có bằng chứng rõ là đã hoán ID).
# Số áo chỉ có nhờ nối qua 1 lần gặp không phân định rõ = 'uncertain'.
ENC_SPLIT = False                       # mặc định tắt: trên clip_3400 không thêm được lần nối đúng nào, độ phủ -1.8 điểm
ENC_R = 1.5                             # mét
ENC_MIN_FRAMES = 3                      # lần gặp phải kéo dài >= N frame (bỏ nhiễu phép chiếu 1-2 frame)
ENC_GAP_FRAMES = 2                      # lấp lỗ <= N frame trong 1 lần gặp
ENC_SWAP_MARGIN = 0.3                   # chuyển động ủng hộ hoán ID khi phương án hoán rẻ hơn "giữ ID" > N (tổng chi phí)
ENC_MOTION_SUSPECT = False              # True: chuyển động ủng hộ hoán ID -> 'uncertain' (clip_3400: 2/2 ca xem lại
                                        # trên video là tracker đúng, chuyển động sai -> mặc định tắt)
ENC_UNMATCHED = 1.0                     # chi phí để 1 cặp A/B không nối với ai (= khoảng cách bằng bán kính cho phép)
