"""Đo thời gian từng bước của pipeline.

    prof = StepTimer(device)
    with prof("legibility", n_items=len(crops)):
        ...
    prof.add("det_inference", ms=12.3)        # thời gian đo sẵn (vd. results.speed của ultralytics)
    prof.report()

Trên GPU, CUDA/MPS chạy bất đồng bộ nên phải synchronize trước khi bấm giờ,
nếu không thời gian sẽ bị tính nhầm sang bước sau.
"""
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class StepTimer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.calls = defaultdict(list)    # step -> [ms mỗi lần gọi]
        self.items = defaultdict(int)     # step -> tổng số item (crop/frame) đã xử lý
        self.order = []

    def _register(self, step):
        if step not in self.calls:
            self.order.append(step)

    @contextmanager
    def __call__(self, step: str, n_items: int = 1):
        self._register(step)
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.calls[step].append((time.perf_counter() - t0) * 1000)
            self.items[step] += n_items

    def add(self, step: str, ms: float, n_items: int = 1):
        self._register(step)
        self.calls[step].append(ms)
        self.items[step] += n_items

    def summary(self, n_frames: int, wall_s: float) -> dict:
        rows = {}
        for step in self.order:
            ms = np.array(self.calls[step])
            n_items = self.items[step]
            rows[step] = {
                "calls": len(ms),
                "items": n_items,
                "total_s": round(ms.sum() / 1000, 3),
                "ms_per_call_mean": round(float(ms.mean()), 2),
                "ms_per_call_p50": round(float(np.median(ms)), 2),   # p50 ít bị ảnh hưởng bởi warm-up
                "ms_per_item": round(float(ms.sum() / max(n_items, 1)), 2),
                "ms_per_frame": round(float(ms.sum() / max(n_frames, 1)), 2),
                "pct_wall": round(100 * ms.sum() / 1000 / max(wall_s, 1e-9), 1),
            }
        return {"device": self.device, "frames": n_frames, "wall_s": round(wall_s, 2),
                "fps": round(n_frames / max(wall_s, 1e-9), 2), "steps": rows}

    @staticmethod
    def print_table(s: dict):
        print(f"\n  ⏱  Thời gian theo bước  (device={s['device']}, {s['frames']} frames, "
              f"{s['wall_s']}s, {s['fps']} fps)")
        hdr = f"  {'bước':<22}{'calls':>7}{'items':>7}{'total s':>9}{'ms/call p50':>13}{'ms/item':>9}{'ms/frame':>10}{'%':>7}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for step, r in s["steps"].items():
            print(f"  {step:<22}{r['calls']:>7}{r['items']:>7}{r['total_s']:>9.2f}{r['ms_per_call_p50']:>13.1f}"
                  f"{r['ms_per_item']:>9.1f}{r['ms_per_frame']:>10.1f}{r['pct_wall']:>6.1f}%")
        print("  (├/└ là chi tiết bên trong '2. detect+track', không cộng thêm vào tổng)")
        if s["device"] != "cuda":
            print("  ⚠ ultralytics chỉ synchronize trên CUDA: trên MPS/CPU 'yolo inference' bị tính thiếu và dồn sang "
                  "'yolo postprocess' — hãy đọc gộp 2 dòng này. Tổng '2. detect+track' vẫn chính xác.")
