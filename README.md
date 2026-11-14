# Player Detection + Jersey Number Recognition

Phát hiện và theo dõi cầu thủ trong video bóng đá, sau đó nhận dạng số áo cho từng cầu thủ.
**Chỉ chạy inference, không cần train.** Weights nhận dạng số áo lấy từ paper
[A General Framework for Jersey Number Recognition in Sports Video](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)
(Koshkina & Elder, CVPRW 2024), fine-tune trên SoccerNet.

```
video ─► YOLO11-pose (detect người + 17 keypoints)
      ─► ByteTrack (gán track id)
      ─► mỗi N frame, với từng cầu thủ:
           legibility classifier (ResNet34) ── không thấy số ─► bỏ qua
           crop thân áo theo vai/hông
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
pip install -q -r requirements.txt
python download_models.py
python run.py --source /kaggle/input/<dataset>/match.mp4 --output /kaggle/working/outputs
```

> Nếu Google Drive báo hết lượt tải weights: tải 2 file trong `config.MODEL_URLS` về máy, upload
> thành 1 Kaggle Dataset, rồi chạy với `MODELS_DIR=/kaggle/input/<dataset> python run.py ...`

## Tham số hay dùng

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--source` | — | video, ảnh, hoặc folder |
| `--max-frames` | 0 (all) | giới hạn số frame để test nhanh |
| `--stride` | 2 | đọc số mỗi N frame |
| `--imgsz` | 1280 | kích thước input YOLO (cầu thủ nhỏ thì để lớn) |
| `--det-model` | `yolo11m-pose.pt` | đổi sang `yolo11x-pose.pt` cho chính xác hơn, hoặc model detect cầu thủ tự train |
| `--classes` | `0` | class id của cầu thủ (đổi khi dùng detector custom) |
| `--tracker` | `bytetrack.yaml` | hoặc `botsort.yaml` |
| `--min-box-height` | 60 | bỏ cầu thủ nhỏ hơn N px |
| `--legibility-th` | 0.5 | ngưỡng "thấy được số áo" |
| `--sum-th` | 1.0 | tổng trọng số tối thiểu để chốt số áo cho track |
| `--no-legibility` | off | bỏ legibility, đọc số mọi cầu thủ (debug) |
| `--save-crops` | off | lưu crop thân áo kèm số đọc được |

Nếu `--det-model` không phải model pose (ví dụ detector tự train), pipeline sẽ tự chạy thêm
`--pose-model` để lấy keypoints, rồi ghép với bbox theo IoU.

## Output (`outputs/<tên video>/`)
- `<tên>_jersey.mp4`: bbox xanh kèm `#số` khi đã nhận ra số áo, xám kèm `idN` khi chưa
- `tracks.json`: mỗi track gồm `jersey_number` (-1 nếu không đọc được), `votes`, `attempts`, `legible`, `str_reads`
- `detections.csv`: `frame, track_id, x1, y1, x2, y2, det_conf, jersey_number`

## Cấu trúc
```
run.py                  # entry point: analyse (pass 1) + render (pass 2)
config.py               # đường dẫn weights, ngưỡng mặc định
download_models.py      # tải weights legibility + PARSeq từ Google Drive
kaggle_notebook.ipynb
pipeline/
  legibility.py         # ResNet34 legibility classifier
  str_recognizer.py     # PARSeq loader + inference
  torso.py              # crop thân áo từ keypoints COCO
  aggregator.py         # bỏ phiếu số áo theo track
  device.py
third_party/parseq/     # code strhub (PARSeq, Apache-2.0), vendor từ jersey-number-pipeline
```

## Giới hạn
- YOLO COCO detect mọi "person": trọng tài, thủ môn, khán giả sát biên cũng được track. Muốn chỉ lấy cầu thủ
  thì dùng detector train trên dữ liệu bóng đá và set `--classes`.
- Không có bước lọc outlier bằng re-ID như paper gốc. Nếu ByteTrack đổi ID giữa 2 cầu thủ thì phiếu bầu của track bị lẫn.
- Số áo cần đủ lớn: góc quay broadcast xa thường chỉ đọc được khi camera zoom gần.

## License
Weights và phương pháp nhận dạng số áo: CC BY-NC 3.0 (của repo gốc), **chỉ dùng phi thương mại**.
PARSeq: Apache-2.0. Ultralytics YOLO: AGPL-3.0.
