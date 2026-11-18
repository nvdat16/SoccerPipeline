# Player Detection + Jersey Number Recognition

Phát hiện và theo dõi cầu thủ trong video bóng đá, sau đó nhận dạng số áo cho từng cầu thủ.
**Chỉ chạy inference, không cần train.** Weights nhận dạng số áo lấy từ paper
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

## Tham số hay dùng

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--source` | — | video, ảnh, hoặc folder |
| `--max-frames` | 0 (all) | giới hạn số frame để test nhanh |
| `--stride` | 2 | đọc số mỗi N frame |
| `--imgsz` | 1280 | kích thước input YOLO (cầu thủ nhỏ thì để lớn) |
| `--det-model` | `yolo26s-soccer.pt` | model detect (Ultralytics). Không có file này thì dùng `yolo11m-pose.pt` (COCO) |
| `--classes` | tự chọn | class cầu thủ, theo id hoặc tên (`player goalkeeper`). Không truyền thì tự lấy các class tên `person` / `player` / `goalkeeper` / `goalie` |
| `--tracker` | `bytetrack.yaml` | hoặc `botsort.yaml` |
| `--min-box-height` | 60 | bỏ cầu thủ nhỏ hơn N px |
| `--legibility-th` | 0.5 | ngưỡng "thấy được số áo" |
| `--sum-th` | 1.0 | tổng trọng số tối thiểu để chốt số áo cho track |
| `--torso-mode` | `bbox` | `pose`: cắt thân áo theo keypoint vai/hông (như paper). `bbox`: cắt theo tỉ lệ bbox, bỏ hẳn model pose (nhanh hơn, không mất cầu thủ vì thiếu keypoint) |
| `--torso-ratio` | `0.22 0.20 0.78 0.52` | vùng thân áo (x1 y1 x2 y2) theo tỉ lệ bbox, dùng với `--torso-mode bbox` |
| `--no-legibility` | off | bỏ legibility, đọc số mọi cầu thủ (debug) |
| `--save-crops` | off | lưu crop thân áo kèm số đọc được |

Với `--torso-mode pose`: nếu `--det-model` không phải model pose (ví dụ `yolo26s-soccer.pt`), pipeline chạy thêm
`--pose-model` (mặc định `yolo26n-pose.pt`) để lấy keypoints rồi ghép với bbox theo IoU.

## Minimap 2D

Đặt từng cầu thủ thấy trong frame vào đúng vị trí (mét) trên sân, vẽ minimap ở góc video, số áo trong chấm.
Video broadcast là **1 feed nhiều góc máy** (wide, close-up, replay, chuyển cảnh) nên phép chiếu được tính lại
**mỗi frame** và reset ở mỗi lần cắt cảnh:

```
frame ─► phát hiện cắt cảnh (histogram HSV) ── cut? ─► reset tracker + reset phép chiếu
      ─► calibration (mặc định PnLCalib, mỗi 5 frame): điểm mốc + vạch sân CÓ TÊN ─► homography ảnh → sân
      ─► kiểm tra: lỗi reprojection, vùng sân nhìn thấy hợp lý, không nhảy so với dự đoán
      ─► frame xen giữa / thiếu điểm mốc ─► nối H bằng optical flow (chuyển động camera, tối đa ~2 s)
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

Trên minimap: chấm trắng có số = cầu thủ đã nhận ra số áo, xám = chưa, vàng = thủ môn, đen = trọng tài; vùng xanh nhạt
= phần sân camera đang nhìn thấy. Frame không phải cảnh sân (close-up, khán đài, replay logo) -> minimap mờ đi, giữ vị
trí cuối tối đa 3 s, ghi "no pitch view". Một cầu thủ ra khỏi khung hình thì biến khỏi minimap (chỉ vẽ ai đang thấy).

Hệ toạ độ: (0, 0) = góc sân trên-trái của minimap, x theo chiều dài (0..L), y theo chiều rộng (0..W); cạnh dưới
minimap là đường biên phía camera chính (cả 2 backend cùng quy ước).

Giới hạn:
- Chưa chia đội (chấm cùng màu cho cả 2 đội) và chưa có bóng.
- Cảnh chỉ thấy vạch biên + cỏ (giữa sân, camera zoom) có ít điểm mốc -> sau ~2 s optical flow thì mất phép chiếu.
- Homography coi sân là mặt phẳng và bỏ qua méo ống kính: ở mép khung hình góc rộng có thể lệch 1–2 m.
- Trong 1 shot, H mới lệch > 5 m so với dự đoán (thường do model nhầm khung thành trái/phải) bị loại và optical flow
  nối qua; chỉ chuyển khi các lần đo trong `PITCH_SWITCH_AFTER` (12) frame liên tiếp cùng chỉ về vị trí mới.
  Nếu ngay đầu shot model đã sai thì phải chờ tới khi nó tự đúng lại.
- Góc máy lạ (tactical cam, sau khung thành) có thể không nhận ra.

## Output (`outputs/<tên video>/`)
- `<tên>_jersey.mp4`: bbox xanh kèm `#số` khi đã nhận ra số áo, xám kèm `idN` khi chưa (+ minimap nếu `--minimap`)
- `tracks.json`: mỗi track gồm `role`, `jersey_number` (-1 nếu không đọc được), `votes`, `attempts`, `legible`,
  `str_reads`; thêm `shots` và (với `--minimap`) thống kê `minimap`
- `detections.csv`: `frame, track_id, x1, y1, x2, y2, det_conf, jersey_number, class, shot_id, x_m, y_m, h_status`
  (`x_m, y_m` thô, chưa làm mượt; `h_status` = `kp` | `flow` | `none`)
- `positions.csv` (`--minimap`): `frame, track_id, role, jersey_number, shot_id, h_status, x_m, y_m` đã làm mượt
- `timing.json`: thời gian suy luận từng bước (decode, detect+track, legibility, pose, crop, PARSeq, vote, render)

## Cấu trúc
```
run.py                  # entry point: analyse (pass 1) + render (pass 2)
scripts/                # setup.sh, run_*.sh cho từng luồng, compare.sh, diagnose.sh
tools/diagnose.py       # chẩn đoán từng bước trên mọi cầu thủ
yolo26s-soccer.pt       # detector bóng đá (0 ball, 1 goalkeeper, 2 player, 3 referee)
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
  pitch.py              # 32 điểm mốc sân -> homography ảnh->sân, kiểm tra, optical flow
  minimap.py            # vẽ minimap, làm mượt vị trí, overlay debug
  calib_pnl.py          # backend PnLCalib (HRNet điểm + vạch) -> homography
  device.py
third_party/parseq/     # code strhub (PARSeq, Apache-2.0), vendor từ jersey-number-pipeline
third_party/pnlcalib/   # PnLCalib, chỉ phần inference (GPL-2.0-or-later)
```

## Giới hạn
- YOLO COCO detect mọi "person": trọng tài, thủ môn, khán giả sát biên cũng được track. Muốn chỉ lấy cầu thủ
  thì dùng detector train trên dữ liệu bóng đá (vd `--det-model yolo26s-soccer.pt`); class cầu thủ được tự chọn theo tên.
- Không có bước lọc outlier bằng re-ID như paper gốc. Nếu ByteTrack đổi ID giữa 2 cầu thủ thì phiếu bầu của track bị lẫn.
- Số áo cần đủ lớn: góc quay broadcast xa thường chỉ đọc được khi camera zoom gần.
- `yolo26s-soccer` gần như **không detect được cầu thủ ở cảnh cận** (box > ~300px) — đúng lúc số áo rõ nhất.
  Dùng `run_coco.sh` cho cảnh cận, hoặc fine-tune thêm ảnh cận.
- PARSeq có xu hướng bịa "10" khi số mờ hoặc crop lẫn logo; legibility giúp chặn phần lớn.

## License
Weights và phương pháp nhận dạng số áo: CC BY-NC 3.0 (của repo gốc), **chỉ dùng phi thương mại**.
PARSeq: Apache-2.0. Ultralytics YOLO: AGPL-3.0. PnLCalib (`third_party/pnlcalib`): code GPL-2.0-or-later, weights
CC BY 4.0. Pitch keypoint model Roboflow `sports`: kiểm tra license của repo đó trước khi dùng ngoài mục đích nghiên cứu.
