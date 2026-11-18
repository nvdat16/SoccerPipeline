# PnLCalib (vendored, chỉ phần inference)

Nguồn: https://github.com/mguti97/PnLCalib — commit `42088864170f75ea02e76d20e9a9a1db59edb3e3`.
Gutiérrez-Pérez & Agudo, "PnLCalib: Sports Field Registration via Points and Lines Optimization"
(kế nhiệm "No Bells, Just Whistles", CVPRW 2024).

Code: GPL-2.0-or-later (xem LICENSE). Weights (`SV_kp`, `SV_lines`, release v1.0.0): CC BY 4.0.
`model/cls_hrnet*.py` dẫn xuất từ HRNet (MIT).

Chỉ chép các file cần cho inference: `model/cls_hrnet.py`, `model/cls_hrnet_l.py`,
`utils/utils_heatmap.py`, `utils/utils_calib.py`, `utils/utils_optimize.py`, `config/hrnetv2_w48*.yaml`.
Thay đổi so với bản gốc:
- đóng gói thành package `pnlcalib` (import `from utils...` -> `from pnlcalib.utils...`);
- bỏ `import matplotlib.pyplot` trong `utils_optimize.py` (không dùng khi inference).
