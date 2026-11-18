"""Backend calibration PnLCalib (Gutiérrez-Pérez & Agudo, kế nhiệm "No Bells, Just Whistles").

Hai HRNet-w48 chạy trên frame 960x540:
  - keypoint model: 57 điểm mốc sân (góc, giao điểm vạch, tiếp điểm vòng tròn, chấm 11m...);
  - line model: 23 vạch sân CÓ TÊN (vạch giữa, vòng cấm trái/phải, vòng 5m50, biên, cột/xà...).
Giao điểm các vạch có tên được bổ sung thành keypoint, rồi FramebyFrameCalib thử RANSAC nhiều ngưỡng
và chọn homography mặt sân có lỗi reprojection nhỏ nhất (tuỳ chọn tinh chỉnh thêm theo vạch).

Vì vạch có tên + được train trên SoccerNet (nhiều cảnh zoom gần vòng cấm), backend này không nhầm
"vạch 16.5m + cung phạt đền" thành "vạch giữa sân + vòng tròn giữa" như model 32 điểm của Roboflow.

Code vendor ở third_party/pnlcalib (GPL-2.0-or-later), weights CC BY 4.0.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PNL_DIR = Path(__file__).resolve().parents[1] / "third_party" / "pnlcalib"
if str(PNL_DIR) not in sys.path:
    sys.path.insert(0, str(PNL_DIR))

from pnlcalib.model.cls_hrnet import get_cls_net  # noqa: E402
from pnlcalib.model.cls_hrnet_l import get_cls_net as get_cls_net_l  # noqa: E402
from pnlcalib.utils.utils_calib import FramebyFrameCalib  # noqa: E402
from pnlcalib.utils.utils_heatmap import (complete_keypoints, coords_to_dict,  # noqa: E402
                                          get_keypoints_from_heatmap_batch_maxpool,
                                          get_keypoints_from_heatmap_batch_maxpool_l)

NET_W, NET_H = 960, 540
# PnLCalib: gốc ở tâm sân, sân 105 x 68. Minimap: gốc ở góc sân -> cộng nửa sân.
TO_PITCH = np.array([[1, 0, 52.5], [0, 1, 34.0], [0, 0, 1]])


def _load(net, path, device):
    state = torch.load(str(path), map_location="cpu", weights_only=True)   # state_dict thuần, không unpickle code
    net.load_state_dict(state)
    return net.to(device).eval()


class PnLCalibModel:
    def __init__(self, kp_weights, line_weights, device, kp_threshold=0.3434, line_threshold=0.7867,
                 refine_lines=False):
        cfg_dir = PNL_DIR / "pnlcalib" / "config"
        cfg = yaml.safe_load(open(cfg_dir / "hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open(cfg_dir / "hrnetv2_w48_l.yaml"))
        self.device = device
        self.model = _load(get_cls_net(cfg), kp_weights, device)
        self.model_l = _load(get_cls_net_l(cfg_l), line_weights, device)
        self.kp_th, self.line_th, self.refine = kp_threshold, line_threshold, refine_lines
        self.cam, self.shape = None, None

    @torch.no_grad()
    def estimate(self, frame):
        """Trả (H ảnh(px) -> sân(m, gốc ở góc sân) | None, info)."""
        h0, w0 = frame.shape[:2]
        if self.cam is None or self.shape != (h0, w0):
            self.cam, self.shape = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True), (h0, w0)
        t0 = time.perf_counter()
        small = cv2.resize(frame, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
        x = x.unsqueeze(0).to(self.device)
        # fp16 trên CUDA (tensor core): kết quả gần như trùng fp32 (đo trên MPS: lệch trung vị 0.17 m)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            heat, heat_l = self.model(x), self.model_l(x)
        heat, heat_l = heat.float(), heat_l.float()
        kp = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool(heat[:, :-1]), threshold=self.kp_th)
        ln = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool_l(heat_l[:, :-1]), threshold=self.line_th)
        info = {"model_ms": (time.perf_counter() - t0) * 1000, "n_kp": len(kp[0]), "n_lines": len(ln[0])}
        kp, ln = complete_keypoints(kp[0], ln[0], w=NET_W, h=NET_H, normalize=True)
        info["n_kp_total"] = len(kp)
        # toạ độ ảnh gốc của các điểm dùng để calibrate (cho --pitch-debug)
        info["kp_xy"] = np.array([[v["x"] * w0, v["y"] * h0] for v in kp.values()]).reshape(-1, 2)
        info["kp_conf"] = np.ones(len(info["kp_xy"]))
        if len(kp) < 4:
            return None, info
        self.cam.update(kp, ln)
        try:
            res = self.cam.heuristic_voting_ground(refine_lines=self.refine)
        except (np.linalg.LinAlgError, ValueError):
            res = None
        if res is None or res.get("homography") is None:
            return None, info
        info["rep_err_px"] = float(res["rep_err"])
        H = TO_PITCH @ res["homography"]
        return H / H[2, 2], info
