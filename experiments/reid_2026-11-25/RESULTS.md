# So sánh model re-ID cho bước nối định danh (2026-11-25)

## Mục tiêu
Khi track của một cầu thủ bị đứt giữa shot, `IdentityLinker` nối đoạn trước (A) với đoạn sau (B) bằng vị trí và chuyển động
trên sân. Nếu có đồng đội đứng gần vị trí dự đoán (đám đông), luật đám đông sẽ chặn cặp nối đó. Re-ID theo ngoại hình
được dùng để phá thế này: nếu cuối A giống B hơn mọi đồng đội đứng gần, chênh lệch lớn hơn `REID_MARGIN` (khoảng cách
cosine), thì cho nối. Thí nghiệm này so sánh 3 checkpoint OSNet cho vai trò đó.

## Model

| Tên trong log | Kiến trúc | Dữ liệu train | Nguồn |
|---|---|---|---|
| `osnet_ain_x1_0_msmt17` | OSNet-AIN x1.0 | ImageNet → MSMT17 (người đi bộ, camera giám sát) | model zoo torchreid |
| `osnet_x1_0_sportsmot_deepeiou` | OSNet x1.0 | ImageNet → Market-1501 → crop cầu thủ SportsMOT (616 ID; bóng đá, bóng rổ, bóng chuyền) | Deep-EIoU (Huang et al., WACVW 2024), `sports_model.pth.tar-60` |
| `osnet_x1_0_soccernet_reid` | OSNet x1.0 | ImageNet → SoccerNet-ReID train | baseline sn-reid (bảng "Pretrained Models", OSNet_x1_0: mAP 83.4, rank-1 78.0) |

Checkpoint SoccerNet-ReID được xác nhận đúng nguồn qua 3 điểm: trường `rank1 = 0.7799` (khớp 78.0 trong bảng),
classifier có 161,443 lớp (ID đánh theo từng pha bóng của SoccerNet-ReID), backbone 2.19M tham số (bảng ghi 2.2M).
Checkpoint lưu ở epoch 10. Cả 3 model dùng input 256×128 và chuẩn hoá ImageNet.

## Dữ liệu và cách đo
- Clip: `clip_3400` = `video1.mp4` từ frame 3400 đến 4899 (1500 frame, 25 fps, 1280×720), lấy từ run `outputs/minimap_v2/clip_3400`.
  Detection và track giữ nguyên, chỉ thay embedding.
- Embedding: mọi box player/GK cao ≥ 40 px, mỗi 2 frame (giống `run.py`), tổng 6,639 crop.
- **Probe (ngoại hình thuần):** `link_bench` cắt nhân tạo các track dài (5 seed). Chỉ xét chỗ đứt "đông", tức B có đồng đội
  cách < 3 m (n = 56; có thêm 23 chỗ đứt bị loại vì thiếu embedding). Query là trung bình embedding 10 frame cuối của A;
  gallery là B (10 frame đầu) và các đồng đội gần đó.
  - rank-1: B là ứng viên gần A nhất.
  - cho qua đúng: B gần hơn mọi đồng đội ít nhất `margin`, tức linker sẽ chấp nhận cặp đúng.
  - cho qua sai: một đồng đội gần hơn mọi ứng viên khác ít nhất `margin`, tức linker có thể nối nhầm.
- **link_bench:** 518 chỗ đứt nhân tạo (5 seed). Linker chạy đầy đủ (global Hungarian, ẩn số áo), có hoặc không có embedding.
- **Chỗ đứt thật:** `relink.py` chạy trên track gốc. Không có đáp án đúng, nên chỉ đếm số cặp và số link.

## Kết quả

### 1. Phân biệt đồng đội thuần ngoại hình (n = 56)

| Model | rank-1 | d(A,B) trung vị | d(A, đồng đội gần nhất) trung vị |
|---|---|---|---|
| MSMT17 | 67.9% | 0.103 | 0.130 |
| **Deep-EIoU SportsMOT** | **89.3%** | 0.118 | 0.219 |
| SoccerNet-ReID | 80.4% | 0.162 | 0.235 |

Theo margin (cho qua đúng / cho qua sai):

| margin | MSMT17 | Deep-EIoU SportsMOT | SoccerNet-ReID |
|---|---|---|---|
| 0.00 | 67.9 / 32.1 | 89.3 / 10.7 | 80.4 / 19.6 |
| 0.03 | 41.1 / 23.2 | **78.6 / 10.7** | 71.4 / 12.5 |
| 0.06 (mặc định) | 19.6 / 10.7 | **71.4 / 10.7** | 55.4 / 5.4 |
| 0.10 | 0.0 / 5.4 | 44.6 / 7.1 | 37.5 / **0.0** |
| 0.15 | 0.0 / 0.0 | 8.9 / 5.4 | **30.4 / 0.0** |

### 2. link_bench (518 chỗ đứt nhân tạo, trong đó 79 chỗ "đông")

| Cấu hình | Nối đúng (tổng) | Nối sai (tổng) | Nối đúng (đông) | Nối sai (đông) | Độ chính xác | Cặp đông được re-ID cho qua |
|---|---|---|---|---|---|---|
| Không re-ID | 387 | 33 | 58 | 8 | 92.1% | — |
| MSMT17 | 380 | 32 | 58 | 8 | 92.2% | 46 |
| **Deep-EIoU SportsMOT** | **392** | **22** | **61** | **0** | **94.7%** | 87 |
| SoccerNet-ReID | 390 | 26 | 57 | 4 | 93.8% | 98 |

Margin khác 0.06:

| Model | margin | Nối đúng | Nối sai | Độ chính xác |
|---|---|---|---|---|
| Deep-EIoU | 0.03 | 392 | 22 | 94.7% |
| Deep-EIoU | 0.10 | 390 | 25 | 94.0% |
| SoccerNet-ReID | 0.03 | 386 | 23 | 94.4% |
| SoccerNet-ReID | 0.10 | 390 | 32 | 92.4% |

### 3. Chỗ đứt thật (`relink.py`, không có đáp án đúng)

| Cấu hình | Cặp đông bị chặn | Re-ID phân định | Link nhờ re-ID | Tổng link | Detection có số nhờ nối |
|---|---|---|---|---|---|
| Không re-ID | 96 | — | — | 47 | 5.2% |
| MSMT17 | 92 | 4 | 2 | 50 | 5.3% |
| Deep-EIoU SportsMOT | 91 | 5 | 2 | 49 | 5.2% |
| SoccerNet-ReID | 88 | 8 | 2 | 48 | 5.0% |

Lý do re-ID gần như không có tác dụng ở đây (phân tích với cả MSMT17 lẫn Deep-EIoU): trong 96 cặp đông, 45 cặp thiếu
embedding (34 cặp thiếu ở B, 11 cặp thiếu ở A). Đoạn B thật rất ngắn: trung vị 6 frame, 45/96 cặp có B dưới 4 frame.
Embedding chỉ tính mỗi 2 frame mà linker cần ít nhất 2 frame. Các cặp còn lại không qua được vì một đồng đội giống A hơn B
(30–32 cặp) hoặc chênh lệch nhỏ hơn margin (14–17 cặp). Vì không có đáp án đúng, chưa biết các lần chặn này là đúng hay sai.

## Kết luận
1. Re-ID huấn luyện trên dữ liệu thể thao tốt hơn rõ rệt re-ID người đi bộ (MSMT17) ở việc phân biệt đồng đội mặc áo giống nhau.
2. Deep-EIoU SportsMOT cho rank-1 cao nhất và kết quả link_bench tốt nhất ở margin 0.03–0.06. SoccerNet-ReID tách xa hơn
   khi đã chắc chắn: ở margin ≥ 0.10 nó không cho qua cặp sai nào mà vẫn cho qua đúng 30–38%. Nó phù hợp khi cần chế độ thận trọng.
3. Trên chỗ đứt thật, cả 3 model đều không cải thiện độ phủ số áo. Nút thắt là đoạn B quá ngắn nên thiếu embedding,
   không phải chất lượng model.

## Hạn chế
- Chỉ 1 clip (1500 frame, 720p). Probe có n = 56; 5 seed cắt trên cùng các track nên không độc lập.
  Chênh lệch vài điểm % (khoảng 3–5 trường hợp) chưa có ý nghĩa thống kê.
- Chỗ đứt nhân tạo dễ hơn chỗ đứt thật: A và B đều dài, và không bị che khuất ở chỗ cắt.
- Ngưỡng margin là khoảng cách tuyệt đối, không chuẩn hoá theo thang khoảng cách riêng của từng model.

## Tái lập
```bash
R=outputs/minimap_v2/clip_3400
python tools/reid_bench.py embed $R --video ../video1.mp4 --offset 3400 \
    --weights models/osnet_ain_x1_0_msmt17.pt models/osnet_x1_0_sportsmot_deepeiou.pt models/osnet_x1_0_soccernet_reid.pt
python tools/reid_bench.py probe $R --emb $R/emb_*.npz [--margin 0.06]
python tools/link_bench.py $R [--emb $R/emb_<model>.npz] [--set REID_MARGIN=0.06]
python tools/relink.py $R [--emb $R/emb_<model>.npz] --out $R/relink_<model>
```
Log gốc: `01_embed.log`, `02_probe.log`, `03_link_bench.log`, `04_relink_real.log`, `05_probe_margin_sweep.log`,
`06_link_bench_margin.log`. Lần chạy đầu của MSMT17 và Deep-EIoU (trước khi có SoccerNet-ReID) cho kết quả giống hệt.
