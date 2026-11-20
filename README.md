# Player Detection + Jersey Number Recognition

Phát hiện và theo dõi cầu thủ trong video bóng đá, sau đó nhận dạng số áo, chia đội và đặt cầu thủ lên minimap 2D.
Pipeline **chỉ chạy inference**. Detector `yolo26s-soccer` được train riêng trên `dataset.zip` của
[DatNguyen05/soccer_dataset](https://huggingface.co/datasets/DatNguyen05/soccer_dataset) (xem [Detector & dataset](#detector--dataset)).
Weights nhận dạng số áo lấy từ paper
[A General Framework for Jersey Number Recognition in Sports Video](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)
(Koshkina & Elder, CVPRW 2024), fine-tune trên SoccerNet.

```
video ─► yolo26s-soccer (detect player + goalkeeper)
      ─► ByteTrack (gán track id)
      ─► mỗi N frame, với từng cầu thủ:
           legibility classifier (ResNet34) ── không thấy số ─► bỏ qua
           crop thân áo theo tỉ lệ bbox   (hoặc theo vai/hông với --torso-mode pose)
           PARSeq đọc số (0-99) + xác suất
      ─► bỏ phiếu theo track (cộng xác suất, ưu tiên số 2 chữ số)
      ─► chia đội: histogram màu thân áo ─► KMeans 2 cụm ─► nhãn đội theo frame
      ─► video có nhãn + tracks.json + detections.csv
```

## Chạy trên Kaggle

1. Tạo Kaggle Notebook, import file [kaggle_notebook.ipynb](kaggle_notebook.ipynb) (File → Import Notebook).
2. Settings: **Accelerator = GPU T4**, **Internet = On**, Add Input là dataset chứa video `.mp4`.
3. Run All.

Hoặc chạy bằng lệnh:
```bash
git clone https://github.com/nvdat16/SoccerPipeline.git
cd SoccerPipeline
bash scripts/setup.sh                                   # pip install + tải weights
bash scripts/run_bbox.sh /kaggle/input/<dataset>/match.mp4 /kaggle/working/outputs
```

## Scripts cho từng luồng

Cú pháp chung: `bash scripts/<script>.sh <video> [output_dir] [flag thêm cho run.py...]`.
Biến môi trường: `MAX_FRAMES` (test nhanh), `DEVICE`, `MODELS_DIR`, `PYTHON`.

| Script | Luồng | Khi nào dùng |
|---|---|---|
| `setup.sh` | cài requirements + tải weights | 1 lần sau khi clone |
| **`run_bbox.sh`** | yolo26s-soccer → legibility → crop theo bbox → PARSeq | **mặc định / khuyến nghị** |
| `run_pose.sh` | yolo26s-soccer → legibility → yolo26n-pose (vai/hông) → PARSeq | giống paper; `POSE_MODEL=...` để đổi model pose |
| `run_coco.sh` | yolo11m-pose COCO (detect + keypoints) | khi cần bắt cầu thủ ở cảnh cận; lẫn khán giả |
| `run_no_legibility.sh` | bbox, bỏ legibility, lưu crop | chỉ để debug — PARSeq bịa số |
| **`run_minimap.sh`** | bbox + **minimap 2D** (vị trí cầu thủ trên sân, có số áo) | xem [Minimap](#minimap-2d) |
| `compare.sh` | chạy các luồng trên + in bảng so sánh | chọn luồng cho video mới; `CASES="bbox pose"` |
| `diagnose.sh` | chạy mọi bước trên mọi cầu thủ, ghi `attempts.csv` | tìm vì sao số rõ mà không nhận; `EVERY`, `START` |

```bash
MAX_FRAMES=750 bash scripts/run_bbox.sh match.mp4 outputs/bbox --save-crops
MAX_FRAMES=2500 bash scripts/compare.sh match.mp4 outputs/compare
START=600 MAX_FRAMES=200 EVERY=5 bash scripts/diagnose.sh match.mp4
```

So sánh trên `liver-mu.mp4` (2500 frame, M1 Pro):

| Luồng | Track có số | Ghi chú |
|---|---|---|
| bbox | 35 | 8/12 số nhận thêm so với pose là đúng; 4 lỗi đều là "#10" |
| pose | 22 | 167 crop bị bỏ vì keypoint vai/hông yếu |
| no_legibility | 109 | 35 track "#10", 17 track "#27" → bịa số |

> Nếu Google Drive báo hết lượt tải weights: tải 2 file trong `config.MODEL_URLS` về máy, upload
> thành 1 Kaggle Dataset, rồi chạy với `MODELS_DIR=/kaggle/input/<dataset> python run.py ...`

## Detector & dataset

Detector bóng đá có 4 class: `player`, `goalkeeper`, `referee`, `ball`. Dataset nằm trên Hugging Face:
[DatNguyen05/soccer_dataset](https://huggingface.co/datasets/DatNguyen05/soccer_dataset).

| File | Nguồn |
|---|---|
| `dataset_cu.zip` | trích từ [SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking) |
| `dataset.zip` | thu thập từ [Roboflow Universe](https://universe.roboflow.com) |

```bash
pip install -U huggingface_hub
huggingface-cli download DatNguyen05/soccer_dataset --repo-type dataset --local-dir data/soccer_dataset
```

Cả 4 cấu hình bên dưới được đánh giá trên **cùng một tập đánh giá**, nên so sánh được giữa hai bộ.

Mỗi bộ có 2 cấu hình train:
- `yolo26s`: train thẳng yolo26s trên dataset.
- `yolo26l (...) ⇒ yolo26s`: **knowledge distillation**. Trước tiên train một teacher `yolo26l` trên bộ chỉ gồm các class
  ghi trong ngoặc (`gk + ball` là goalkeeper và ball, `ball` là chỉ ball). Sau đó distill teacher này sang student `yolo26s`.

**`yolo26s-soccer.pt` trong repo** là cấu hình `yolo26s + new dataset`, tức yolo26s train trên `dataset.zip`, không
distillation. Đây là cấu hình có mAP50:95 (`all`) cao nhất: 0.514.

### Kết quả trên `dataset_cu.zip` (SoccerNet Tracking)

`yolo26s`:

| Class | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---|---|---|---|---|
| all | 0.807 | 0.797 | 0.802 | 0.790 | 0.465 |
| player | 0.968 | 0.956 | 0.962 | 0.974 | 0.630 |
| goalkeeper | 0.848 | 0.844 | 0.846 | 0.871 | 0.519 |
| referee | 0.794 | 0.956 | 0.867 | 0.906 | 0.566 |
| ball | 0.619 | 0.443 | 0.516 | 0.407 | 0.145 |

`yolo26l (gk + ball) ⇒ yolo26s`:

| Class | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---|---|---|---|---|
| all | 0.839 | 0.812 | 0.825 | 0.803 | 0.446 |
| player | 0.952 | 0.937 | 0.944 | 0.960 | 0.579 |
| goalkeeper | 0.911 | 0.901 | 0.906 | 0.904 | 0.495 |
| referee | 0.849 | 0.911 | 0.879 | 0.907 | 0.536 |
| ball | 0.643 | 0.501 | 0.563 | 0.443 | 0.173 |

### Kết quả trên `dataset.zip` (Roboflow)

`yolo26s + new dataset` — **đang dùng (`yolo26s-soccer.pt`)**:

| Class | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---|---|---|---|---|
| all | 0.846 | 0.824 | 0.835 | 0.830 | 0.514 |
| player | 0.935 | 0.938 | 0.936 | 0.961 | 0.673 |
| goalkeeper | 0.830 | 0.839 | 0.834 | 0.849 | 0.553 |
| referee | 0.862 | 0.841 | 0.851 | 0.870 | 0.569 |
| ball | 0.757 | 0.677 | 0.715 | 0.638 | 0.261 |

`yolo26l (ball) ⇒ yolo26s + new dataset`:

| Class | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---|---|---|---|---|
| all | 0.853 | 0.802 | 0.827 | 0.828 | 0.507 |
| player | 0.942 | 0.931 | 0.936 | 0.958 | 0.660 |
| goalkeeper | 0.858 | 0.804 | 0.830 | 0.860 | 0.550 |
| referee | 0.849 | 0.828 | 0.838 | 0.849 | 0.554 |
| ball | 0.762 | 0.647 | 0.700 | 0.645 | 0.262 |

Nhận xét:
- Đổi dữ liệu train từ `dataset_cu` sang `dataset.zip` (cùng `yolo26s`, không distill) giúp nhiều nhất:
  mAP50:95 (`all`) 0.465 → 0.514, `ball` tăng mạnh (mAP50 0.407 → 0.638, recall 0.443 → 0.677), `player`
  0.630 → 0.673. Đổi lại, recall của `referee` giảm 0.956 → 0.841 và mAP50 của `goalkeeper` giảm 0.871 → 0.849.
- `ball` là class khó nhất ở cả hai bộ: mAP50:95 chỉ 0.145–0.262, trong khi `player` đạt 0.58–0.67.
- Trên `dataset_cu`, distill từ teacher `yolo26l (gk + ball)` tăng mAP50 của `goalkeeper` (0.871 → 0.904) và `ball` (0.407 → 0.443).
  Đổi lại, mAP50:95 giảm ở `player` (0.630 → 0.579) và ở `all` (0.465 → 0.446): box kém khít hơn.
- Trên `dataset.zip`, distill từ teacher `yolo26l (ball)` gần như không đổi gì: mAP50 của `ball` 0.638 → 0.645,
  mAP50:95 của `all` 0.514 → 0.507, recall của `all` giảm 0.824 → 0.802. Vì vậy bản đang dùng không distill.

## Tham số hay dùng

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--source` | — | video, ảnh, hoặc folder |
| `--max-frames` | 0 (all) | giới hạn số frame để test nhanh |
| `--stride` | 2 | đọc số mỗi N frame |
| `--imgsz` | 1280 | kích thước input YOLO (cầu thủ nhỏ thì để lớn) |
| `--det-model` | `yolo26s-soccer.pt` | model detect (Ultralytics). Không có file này thì dùng `yolo11m-pose.pt` (COCO) |
| `--classes` | tự chọn | class cầu thủ, theo id hoặc tên (`player goalkeeper`). Không truyền thì tự lấy các class tên `person` / `player` / `goalkeeper` / `goalie` |
| `--tracker` | `trackers/bytetrack_soccer.yaml` | ByteTrack với ngưỡng cho bóng đá. `trackers/botsort_soccer.yaml` = thêm bù chuyển động camera; `bytetrack.yaml` = mặc định ultralytics (xem [Cầu thủ ở xa](#cầu-thủ-ở-xa)) |
| `--conf` | 0.1 | ngưỡng **detector**. Để thấp: box 0.1–0.3 không tạo track mới, chỉ giữ track cũ (bước ghép thứ 2 của BYTE). Ngưỡng tạo track nằm trong file yaml |
| `--tile` / `--no-tile` | off | detect thêm trên lưới tile vùng xa camera để bắt cầu thủ nhỏ (`TILE_*` trong `config.py`, xem [Cầu thủ ở xa](#cầu-thủ-ở-xa)) |
| `--min-box-height` | 60 | bỏ cầu thủ nhỏ hơn N px |
| `--legibility-th` | 0.5 | ngưỡng "thấy được số áo" |
| `--sum-th` | 1.0 | tổng trọng số tối thiểu để chốt số áo cho track |
| `--torso-mode` | `bbox` | `pose`: cắt thân áo theo keypoint vai/hông (như paper). `bbox`: cắt theo tỉ lệ bbox, bỏ hẳn model pose (nhanh hơn, không mất cầu thủ vì thiếu keypoint) |
| `--torso-ratio` | `0.22 0.20 0.78 0.52` | vùng thân áo (x1 y1 x2 y2) theo tỉ lệ bbox, dùng với `--torso-mode bbox` |
| `--no-legibility` | off | bỏ legibility, đọc số mọi cầu thủ (debug) |
| `--save-crops` | off | lưu crop thân áo kèm số đọc được |
| `--teams` / `--no-teams` | on | chia 2 đội theo màu áo (xem [Chia đội](#chia-đội)) |
| `--link` / `--no-link` | on | (cần `--minimap`) nối track thành cầu thủ, truyền số áo (xem [Định danh cầu thủ](#định-danh-cầu-thủ-trên-minimap)) |
| `--enc-split` / `--no-enc-split` | off | (cần `--link`) xử lý riêng lúc 2 đồng đội gặp nhau (xem [Cầu thủ ở xa](#cầu-thủ-ở-xa)) |
| `--link-solver` | `global` | (cần `--link`) `global`: ghép Hungarian tối ưu toàn cục; `greedy`: tham lam (bản cũ) |
| `--reid` / `--no-reid` | off | (cần `--link`) ngoại hình OSNet phân định cặp nối trong đám đông (xem [Cầu thủ ở xa](#cầu-thủ-ở-xa)) |

Với `--torso-mode pose`: nếu `--det-model` không phải model pose (ví dụ `yolo26s-soccer.pt`), pipeline chạy thêm
`--pose-model` (mặc định `yolo26n-pose.pt`) để lấy keypoints rồi ghép với bbox theo IoU.

## Chia đội

Không cần model hay train. Chạy mặc định, tắt bằng `--no-teams`. Tham số `TEAM_*` nằm trong `config.py`.

1. Mỗi `--stride` frame, với mỗi cầu thủ cao từ 30 px: lấy vùng thân áo (`TEAM_BBOX_RATIO`) rồi tính histogram
   HSV 21 chiều (18 bin hue cho pixel có màu, 3 bin đen/xám/trắng), bỏ pixel cỏ.
2. Hết video: KMeans 2 cụm trên mẫu của mọi track `player` (tối đa 30 mẫu/track). Đội A là cụm đông hơn.
3. Gán đội **cho từng mẫu**: mẫu xa cả 2 tâm cụm (median + 4·MAD) bị coi là màu lạ, ví dụ trọng tài bị detect
   thành player. Sau đó làm mượt trong track (đa số ±4 mẫu). Mỗi frame lấy nhãn của mẫu gần nhất.
   Gán theo frame vì ByteTrack hay **đổi ID** giữa 2 cầu thủ khác đội khi họ chạm nhau. `team_switches` trong
   `tracks.json` đếm số lần như vậy.
4. Track mà detector gọi là `goalkeeper` nhưng áo khớp màu một đội (≥ 70% mẫu) thực ra là cầu thủ bị nhận nhầm
   class: đổi role thành `player` và gán đội theo màu.

Video: bbox và chấm minimap tô bằng **màu áo thật** của đội, nhãn `A #10` / `B id42`. Thủ môn thật hiện `GK` màu vàng.

Trên `liver-mu` (clip 1500 frame, Liverpool đỏ / MU trắng): 2 cụm "đỏ + cam" / "trắng + xám". 94% detection
cầu thủ có đội. Các track bị đổi ID (trắng → đỏ giữa chừng) được tách đúng theo frame.

Giới hạn:
- **Thủ môn thật không có đội** (`team = -1`). Đã thử gán theo vị trí (cầu thủ sâu nhất, trọng tâm đội, số
  cầu thủ gần khung thành) nhưng đều sai khi một đội tấn công liên tục trong clip.
- Hai đội mặc áo gần màu nhau (ví dụ đỏ và cam, xanh đậm và đen) thì histogram dễ lẫn. Khi đó cần embedding
  (SigLIP/CLIP).
- Áo xanh lá: pixel cỏ bị bỏ, nên chỉ nhận được nếu hơn 70% crop là màu xanh lá.
- Hai cầu thủ chồng nhau: crop thân áo lẫn màu của người phía sau.

## Minimap 2D

Đặt từng cầu thủ thấy trong frame vào đúng vị trí (mét) trên sân, vẽ minimap ở góc video, số áo trong chấm.
Video broadcast là **1 feed nhiều góc máy** (wide, close-up, replay, chuyển cảnh) nên phép chiếu được tính lại
**mỗi frame** và reset ở mỗi lần cắt cảnh:

```
frame ─► phát hiện cắt cảnh (histogram HSV) ── cut? ─► reset tracker + reset phép chiếu
      ─► calibration (mặc định PnLCalib, mỗi 5 frame): điểm mốc + vạch sân CÓ TÊN ─► homography ảnh → sân
      ─► kiểm tra: lỗi reprojection, vùng sân nhìn thấy hợp lý, không nhảy so với dự đoán
      ─► frame xen giữa / thiếu điểm mốc ─► nối H bằng optical flow (chuyển động camera, tối đa ~2 s)
      ─► hết video: frame vẫn thiếu H trong shot ─► lan NGƯỢC từ lần đo tốt kế tiếp qua chuyển động camera
      ─► chân cầu thủ (đáy-giữa bbox) × H ─► (x, y) mét ─► làm mượt theo track ─► minimap + positions.csv
```

Hai backend calibration (`--calib`):

| | `pnlcalib` (mặc định) | `roboflow` |
|---|---|---|
| Model | [PnLCalib](https://github.com/mguti97/PnLCalib): 2 HRNet-w48, 57 điểm mốc + 23 vạch có tên, train SoccerNet | YOLOv8x-pose 32 điểm ([roboflow/sports](https://github.com/roboflow/sports)) |
| Cảnh zoom gần vòng cấm | đúng | **sai**: nhầm vạch 16.5 m + cung phạt đền thành vạch giữa sân + vòng tròn giữa -> cầu thủ gần khung thành bị đặt ra giữa sân |
| Tốc độ | ~0.8 s/lần trên M1 (MPS); nhanh hơn nhiều trên T4 (fp16) -> mặc định đo mỗi 5 frame | ~80 ms/lần trên M1, đo mọi frame |
| Weights | `pnlcalib_SV_kp.pth` + `pnlcalib_SV_lines.pth` (~265 MB mỗi file, GitHub release) | `football-pitch-detection.pt` (140 MB, Google Drive) |

So sánh trên `video1.mp4` (đoạn 60 s tấn công khung thành phải): thủ môn đứng sát khung thành phải, PnLCalib đặt ở
x = 102.3 m (vạch cầu môn ở 105 m), Roboflow đặt ở x = 67.5 m. 7/8 frame kiểm tra gần vòng cấm Roboflow chiếu sai.

```bash
MAX_FRAMES=1500 bash scripts/run_minimap.sh match.mp4 outputs/minimap --pitch-debug   # kiểm tra trước
bash scripts/run_minimap.sh match.mp4 outputs/minimap --minimap-video
```

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--minimap` | off | bật minimap (`download_models.py` tự tải weights) |
| `--calib` | `pnlcalib` | `pnlcalib` hoặc `roboflow` (xem bảng trên) |
| `--pnl-refine` | off | PnLCalib: tinh chỉnh homography theo vạch sân (chậm hơn một chút) |
| `--pitch-every N` | 5 (pnlcalib) / 1 (roboflow) | chạy calibration mỗi N frame, giữa các lần nối bằng optical flow |
| `--pitch-size L W` | `105 68` | chỉ cho `roboflow`; PnLCalib dùng sân 105 x 68 cố định |
| `--pitch-kpt-conf` | 0.5 | chỉ cho `roboflow`: ngưỡng confidence điểm mốc |
| `--pitch-debug` | off | vẽ vạch sân chiếu ngược + điểm mốc lên video: vạch tím phải trùng vạch thật |
| `--reset-on-cut` | bật khi `--minimap` | reset tracker ở cắt cảnh (ID không bị kéo sang góc máy khác). `--no-reset-on-cut` để tắt |
| `--minimap-video` | off | xuất thêm `<tên>_minimap.mp4` chỉ có minimap |
| `--minimap-pos` | `bottom-right` | vị trí minimap: `bottom-left/center/right`, `top-left/center/right` |

Trên minimap: chấm tô màu áo của đội (xem [Chia đội](#chia-đội)), có số nếu đã nhận ra số áo. Cầu thủ chưa rõ đội:
trắng nếu có số, xám nếu chưa. Thủ môn màu vàng, trọng tài màu đen.
Vùng xanh nhạt là phần sân camera đang nhìn thấy. Frame không phải cảnh sân (close-up, khán đài, replay logo) -> minimap mờ đi, giữ vị
trí cuối tối đa 3 s, ghi "no pitch view". Một cầu thủ ra khỏi khung hình thì biến khỏi minimap (chỉ vẽ ai đang thấy).

Hệ toạ độ: (0, 0) = góc sân trên-trái của minimap, x theo chiều dài (0..L), y theo chiều rộng (0..W); cạnh dưới
minimap là đường biên phía camera chính (cả 2 backend cùng quy ước).

Giới hạn:
- Chưa vẽ bóng lên minimap (detector có class `ball` nhưng pipeline chưa dùng).
- Cảnh chỉ thấy vạch biên + cỏ (giữa sân, camera zoom) có ít điểm mốc -> sau ~2 s optical flow thì mất phép chiếu.
  Sau cả video, đoạn thiếu H trong shot được lan ngược từ lần đo tốt kế tiếp (`h_status = fill`). Trên `clip_3400`:
  +2.0% frame, cầu thủ chiếu lên sân được 91.2% -> 94.1% (ví dụ frame 777–798: 1 lần đo lệch làm mất H 22 frame
  giữa cảnh rộng). Ở chỗ chuỗi lan xuôi gặp chuỗi lan ngược, chấm có thể nhảy vài mét nếu 2 lần đo lệch nhau
  (frame 777: 4.7 m); bước nhảy này trước đây vẫn có, chỉ bị che bởi khoảng trống.
- Homography coi sân là mặt phẳng và bỏ qua méo ống kính: ở mép khung hình góc rộng có thể lệch 1–2 m.
- Trong 1 shot, H mới lệch > 5 m so với dự đoán (thường do model nhầm khung thành trái/phải) bị loại và optical flow
  nối qua; chỉ chuyển khi các lần đo trong `PITCH_SWITCH_AFTER` (12) frame liên tiếp cùng chỉ về vị trí mới.
  Nếu ngay đầu shot model đã sai thì phải chờ tới khi nó tự đúng lại.
- Góc máy lạ (tactical cam, sau khung thành) có thể không nhận ra.

## Định danh cầu thủ trên minimap

Track id của ByteTrack sống rất ngắn: trên `clip_3400`, một nửa số track kéo dài chưa tới 1 s, vì bị che khuất, cắt
cảnh hoặc đổi ID. Số áo đọc được ở một track vì vậy bị mất khi cầu thủ sang track mới. `--link` (mặc định bật cùng
`--minimap`, code trong `pipeline/identity.py`, tham số `LINK_*` trong `config.py`) giải quyết bằng cách:

1. **Ghi nhớ khi nhìn rõ**: đoạn track nào đọc được số áo (bỏ phiếu như bình thường) thì gắn số đó.
   Track bị cắt thành nhiều đoạn ở chỗ đội đổi giữa chừng (ByteTrack đổi ID giữa 2 cầu thủ khác đội).
2. **Nối lại khi mất dấu**, dựa trên vị trí (mét) trên minimap. Đoạn A kết thúc, đoạn B cùng đội bắt đầu trong
   vòng 3 s:
   - vị trí dự đoán = cuối A + **vận tốc** A × khoảng trống (ngoại suy tối đa 1 s);
   - cho phép lệch 2 m + 3 m mỗi giây mất dấu (tối đa 10 m);
   - chi phí cộng thêm nếu **hướng chạy** của A và B lệch nhau, hoặc vận tốc đổi đột ngột;
   - **không nối** nếu: có cầu thủ cùng đội khác cũng đứng gần vị trí dự đoán (đám đông), chi phí quá cao
     (`LINK_ACCEPT`), cặp mập mờ, hoặc hai bên đã đọc được số khác nhau;
   - **ghép toàn cục** (`--link-solver global`, mặc định): chọn tập cặp nối có tổng chi phí tốt nhất (Hungarian
     trên từng cụm ứng viên) thay vì lấy cặp rẻ nhất trước. Cặp mập mờ = bỏ nó đi thì phương án tốt nhất còn lại
     chỉ kém hơn < `LINK_AMBIGUITY`; khác bản tham lam ở chỗ có tính tới việc ứng viên thứ hai đã thuộc về đoạn khác.
3. **Định danh = (đội, số áo)**, ví dụ `A26`, `B10`. Số áo đọc ở một đoạn được truyền sang mọi đoạn trong chuỗi.
   Một cầu thủ không thể ở hai nơi cùng lúc: nếu hai chuỗi cùng `A27` xuất hiện trong cùng frame, chuỗi ít phiếu
   hơn bị bỏ số (thường do PARSeq đọc nhầm).

Video: `A #26` là số tự đọc được ở đoạn đó, `A #26*` là số có được nhờ nối, `A #26?` là số chỉ có nhờ nối qua một
lần gặp đồng đội không phân định được (có thể nhầm người). Minimap ghi số trong chấm cho cả ba.

Ràng buộc cứng khi nối: hai chuỗi từng có mặt trong **cùng một frame** là hai người khác nhau, không bao giờ nối.

Trên `clip_3400` (1500 frame, 18 shot), tracker cũ: 284 đoạn, 52 lần nối, 155 cặp bị bỏ vì đám đông. 19 cầu thủ
có số. Không có `player_id` nào xuất hiện hai lần trong cùng một frame. Detection có số áo: 38.6% tự đọc + 5.1% nhờ
nối. Với tracker mặc định mới (`bytetrack_soccer.yaml`): 246 đoạn, 45 lần nối, 110 cặp bỏ vì đám đông, 40.6% + 5.3%.
Trước đây con số là 44% theo cả track, nhưng gồm cả các đoạn bị gán sai số sau khi ByteTrack đổi ID.
Ghép toàn cục + lấp phép chiếu: 246 đoạn, 47 lần nối, 96 cặp bỏ vì đám đông, 40.6% + 5.2%.

Chất lượng nối đo bằng `tools/link_bench.py`: cắt nhân tạo các track dài (518 chỗ đứt có đáp án, 5 seed), ẩn số
áo, đếm số chỗ được nối lại đúng / nối sai:

| Cấu hình | Nối lại đúng | Nối sai | Độ chính xác | Dữ liệu thật: số áo nhờ nối |
|---|---|---|---|---|
| tham lam, ambiguity 0.15, crowd 1.0 (cũ) | 56% | 48 | 85.8% | 5.3% |
| **toàn cục, accept 1.2, ambiguity 0.3, crowd 0.5** (mặc định) | 72% | 37 | 91.0% | 5.2% |
| toàn cục, accept 1.0, ambiguity 0.3, crowd 0.5 | 70% | 26 | 93.3% | 4.1% |

Chỗ đứt thật hay xảy ra lúc bị che khuất nên khó hơn chỗ cắt nhân tạo: bảng này dùng để so sánh các cấu hình.

Giới hạn:
- Chỉ nối được khi có phép chiếu. Cảnh close-up, replay và khán đài (khoảng 40% frame của clip) làm đứt chuỗi.
- Ở đám đông (phạt góc, tranh chấp trong vòng cấm), cách nối này chủ động không nối. Re-ID ngoại hình chung
  (OSNet, `--reid`) đã thử và không đủ (xem [Cầu thủ ở xa](#cầu-thủ-ở-xa)); cần model re-ID train riêng cho bóng đá.
- Số áo sai từ PARSeq (ví dụ 18 → 28, 11 → 10) vẫn lan ra cả chuỗi. Bước "không ở hai nơi cùng lúc" chỉ bắt được
  khi hai người cùng số xuất hiện cùng lúc.

## Cầu thủ ở xa

Nguyên tắc: **đọc số khi gần, giữ danh tính bằng chuyển động khi xa, lan số áo cả tiến lẫn lùi** (offline,
union-find trong `identity.py`). Đo bằng `tools/far_report.py`, thử tham số định danh bằng `tools/relink.py`:

```bash
python tools/far_report.py outputs/minimap/match                          # A-E: chỉ đọc CSV
python tools/far_report.py outputs/minimap/match --video match.mp4        # + F: recall cầu thủ nhỏ (full vs tile)
python tools/relink.py outputs/minimap/match --set LINK_AMBIGUITY=0.2     # chạy lại định danh, không cần GPU
```

`far_report` in theo nhóm chiều cao box: số lần track đứt giữa shot / 1000 detection, độ phủ số áo
(`read` / `linked` / `uncertain`), số lần 2 đồng đội gặp nhau (< 1.5 m) và bao nhiêu lần có track đứt ngay sau đó,
track đọc ra 2 số áo ở 2 quãng tách biệt (dấu hiệu hoán ID), và recall cầu thủ nhỏ.

Kết quả trên `clip_3400` (1500 frame, 720p), từ đó chọn mặc định:

| Thay đổi | Kết quả | Mặc định |
|---|---|---|
| Detector `conf` 0.3 → 0.1, tracker `bytetrack_soccer.yaml` | Với conf 0.3 bước ghép thứ 2 của BYTE (box 0.1–0.3 giữ track cũ) không bao giờ nhận được box nào. Sửa: đứt track giữa shot 10.5 → 7.5 / 1000 detection (−29%), "gặp nhau rồi đứt" 58% → 39%, số áo phủ 43.9% → 45.8% | **bật** |
| BoT-SORT + bù chuyển động camera (`botsort_soccer.yaml`) | Đứt track như trên (7.5), số áo phủ 38.1%, tracker chậm ~3 lần | tắt |
| Cannot-link: 2 chuỗi có mặt trong cùng 1 frame không được nối | Chặn được lần nối sai (2 cầu thủ áo trắng đứng cạnh nhau); trên dữ liệu không có `--enc-split` thì chưa gặp ca nào | **luôn bật** |
| `--enc-split`: cắt track lúc 2 đồng đội gặp nhau, phân định lại bằng chuyển động | Xem lại trên video: 2/2 lần "chuyển động nói đã hoán ID" đều sai (2 người chồng nhau chỉ cách ~0.5 m trên sân, trong nhiễu phép chiếu) -> chuyển sang tin tracker; khi đó không thêm được lần nối đúng nào, độ phủ −1.8 điểm | tắt |
| `--tile`: detect thêm trên tile vùng xa | 80 frame rải khắp `video1.mp4`: full frame (imgsz 1280) thấy 97 cầu thủ < 45 px, tile thêm 0 | tắt |
| Ghép toàn cục (Hungarian) thay tham lam | Benchmark cắt track: nối lại đúng 56% -> 72%, nối sai 48 -> 37; dữ liệu thật giữ 5.2% số áo nhờ nối | **bật** |
| Lan ngược phép chiếu (`fill`) | +2.0% frame có H, cầu thủ chiếu lên sân 91.2% -> 94.1% | **luôn bật** |
| `--reid`: OSNet-AIN (MSMT17) phân định cặp bị luật đám đông chặn | Mô phỏng đứt track: trong đám đông ngoại hình chọn đúng 62% (vị trí: 39%), chỉ quyết khi chênh rõ thì ~79% nhưng chỉ quyết được 12–21% ca; LDA học theo video không giúp. Benchmark: +5 nối đúng / +3 nối sai; chạy thật: nối thêm 1 lần; chậm thêm ~8% thời gian (5 ms/crop trên M1). Đồng đội mặc áo giống hệt, 720p không đủ chi tiết | tắt |

Chưa làm: tracker trong hệ toạ độ sân (Kalman theo mét). Sau các thay đổi trên, 65 lần track đứt giữa shot chưa
nối được, linker bỏ 110 cặp vì có đồng đội đứng gần: đúng những ca vị trí + vận tốc không phân biệt được, nên một
tracker cũng chỉ dùng vị trí + vận tốc sẽ không giải quyết được. Cần thêm tín hiệu khác (số áo đọc được nhiều hơn,
hoặc re-ID ngoại hình cho khoảng trống ngắn — OSNet chung đã thử, không đủ; cần model train trên cầu thủ, ví dụ
SoccerNet Re-ID).

Video độ phân giải cao hơn (1080p, camera rộng hơn) có thể cho kết quả khác: chạy `far_report --video` trước khi
quyết định bật `--tile`.

## Output (`outputs/<tên video>/`)
- `<tên>_jersey.mp4`: bbox màu đội kèm `A #số` / `B idN` (`*` sau số = số có nhờ nối, `?` = không chắc). Track chưa rõ đội: xanh nếu
  có số, xám nếu chưa.
  Thủ môn: `GK` màu vàng. Thêm minimap nếu bật `--minimap`.
- `tracks.json`: mỗi track gồm `role`, `team` (đội chiếm đa số, 0 = A | 1 = B | -1), `team_conf`, `team_switches`,
  `jersey_number` (-1 nếu không đọc được), `votes`, `attempts`, `legible`, `str_reads`. Thêm `shots`, `teams`
  (màu BGR và màu chủ đạo của mỗi đội), và thống kê `minimap` nếu bật `--minimap`. Với `--link`: mỗi track có thêm
  `player_ids`; thêm `players` (mỗi cầu thủ: đội, số, các track, số frame) và thống kê `link`.
- `reads.csv`: mọi lần PARSeq đọc số: `frame, track_id, label, prob`.
- `detections.csv`: `frame, track_id, x1, y1, x2, y2, det_conf, jersey_number, number_source, player_id, class, team,
  shot_id, x_m, y_m, h_status`. `team`, `jersey_number` và `player_id` tính theo frame. `number_source` = `read`
  (đoạn này có đọc ra số đó) | `linked` (nhờ nối) | `uncertain` (chỉ nhờ nối qua lần gặp đồng đội không phân định
  được, hoặc nhờ `--reid`) | rỗng. `x_m, y_m` là giá trị thô, chưa làm mượt. `h_status` = `kp` | `flow` | `fill`
  (lan ngược sau cả video) | `none`.
- `positions.csv` (`--minimap`): `frame, track_id, player_id, role, team, jersey_number, number_source, shot_id,
  h_status, x_m, y_m`, đã làm mượt
- `timing.json`: thời gian suy luận từng bước (decode, detect+track, legibility, pose, crop, PARSeq, vote, render)

## Cấu trúc
```
run.py                  # entry point: analyse (pass 1) + render (pass 2)
scripts/                # setup.sh, run_*.sh cho từng luồng, compare.sh, diagnose.sh
tools/diagnose.py       # chẩn đoán từng bước trên mọi cầu thủ
tools/far_report.py     # đo tracking/định danh theo kích thước box, lần gặp đồng đội, recall box nhỏ (tile)
tools/relink.py         # chạy lại bước định danh từ CSV (không GPU) để thử tham số
tools/link_bench.py     # benchmark bước nối có đáp án: cắt nhân tạo track dài, đếm nối lại đúng / sai
trackers/               # cấu hình tracker (bytetrack_soccer.yaml mặc định, botsort_soccer.yaml thêm GMC)
yolo26s-soccer.pt       # detector bóng đá (0 ball, 1 goalkeeper, 2 player, 3 referee), xem "Detector & dataset"
config.py               # đường dẫn weights, ngưỡng mặc định
download_models.py      # tải weights legibility + PARSeq từ Google Drive
kaggle_notebook.ipynb
pipeline/
  legibility.py         # ResNet34 legibility classifier
  str_recognizer.py     # PARSeq loader + inference
  torso.py              # crop thân áo: theo keypoints COCO hoặc theo tỉ lệ bbox
  profiler.py           # đo thời gian từng bước -> timing.json
  aggregator.py         # bỏ phiếu số áo theo track
  shots.py              # phát hiện cắt cảnh
  pitch.py              # 32 điểm mốc sân -> homography ảnh->sân, kiểm tra, optical flow, lan ngược lấp khoảng trống
  minimap.py            # vẽ minimap, làm mượt vị trí, overlay debug
  team.py               # chia đội: histogram màu áo + KMeans, nhãn đội theo frame
  identity.py           # nối track thành cầu thủ theo vị trí/vận tốc/hướng trên minimap (Hungarian), truyền số áo
  reid.py               # OSNet-AIN (rút gọn từ torchreid, MIT) -> embedding ngoại hình cho --reid
  tiling.py             # detect trên tile (cầu thủ nhỏ) + gọi tracker trực tiếp với detection đã gộp
  calib_pnl.py          # backend PnLCalib (HRNet điểm + vạch) -> homography
  device.py
third_party/parseq/     # code strhub (PARSeq, Apache-2.0), vendor từ jersey-number-pipeline
third_party/pnlcalib/   # PnLCalib, chỉ phần inference (GPL-2.0-or-later)
```

## Giới hạn
- YOLO COCO detect mọi "person": trọng tài, thủ môn, khán giả sát biên cũng được track. Muốn chỉ lấy cầu thủ
  thì dùng detector train trên dữ liệu bóng đá (vd `--det-model yolo26s-soccer.pt`); class cầu thủ được tự chọn theo tên.
- Không có bước lọc outlier bằng re-ID như paper gốc. ByteTrack đổi ID giữa 2 cầu thủ **khác đội** thì được tách ra
  (chia đội + `--link`). Đổi ID giữa 2 cầu thủ **cùng đội** thì phiếu bầu số áo vẫn bị lẫn.
- Số áo cần đủ lớn: góc quay broadcast xa thường chỉ đọc được khi camera zoom gần.
- `yolo26s-soccer` gần như **không detect được cầu thủ ở cảnh cận** (box > ~300px) — đúng lúc số áo rõ nhất.
  Dùng `run_coco.sh` cho cảnh cận, hoặc fine-tune thêm ảnh cận.
- PARSeq có xu hướng bịa "10" khi số mờ hoặc crop lẫn logo; legibility giúp chặn phần lớn.

## License
Weights và phương pháp nhận dạng số áo: CC BY-NC 3.0 (của repo gốc), **chỉ dùng phi thương mại**.
PARSeq: Apache-2.0. Ultralytics YOLO: AGPL-3.0. PnLCalib (`third_party/pnlcalib`): code GPL-2.0-or-later, weights
CC BY 4.0. Pitch keypoint model Roboflow `sports`: kiểm tra license của repo đó trước khi dùng ngoài mục đích nghiên cứu.
OSNet (`pipeline/reid.py`, chỉ dùng với `--reid`): kiến trúc từ deep-person-reid/torchreid (MIT); weights train trên
MSMT17, bộ dữ liệu chỉ cho phép nghiên cứu.
