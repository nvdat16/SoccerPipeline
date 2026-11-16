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

## Output (`outputs/<tên video>/`)
- `<tên>_jersey.mp4`: bbox xanh kèm `#số` khi đã nhận ra số áo, xám kèm `idN` khi chưa
- `tracks.json`: mỗi track gồm `jersey_number` (-1 nếu không đọc được), `votes`, `attempts`, `legible`, `str_reads`
- `detections.csv`: `frame, track_id, x1, y1, x2, y2, det_conf, jersey_number`
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
  device.py
third_party/parseq/     # code strhub (PARSeq, Apache-2.0), vendor từ jersey-number-pipeline
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
PARSeq: Apache-2.0. Ultralytics YOLO: AGPL-3.0.
