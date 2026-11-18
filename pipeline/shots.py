"""Phát hiện cắt cảnh (shot boundary) trong feed broadcast nhiều góc máy.

So sánh histogram màu HSV (Hue x Saturation) của frame thu nhỏ:
  - cắt cứng: khoảng cách Bhattacharyya giữa 2 frame liền kề vượt ngưỡng tuyệt đối VÀ lớn hơn nhiều
    lần mức thay đổi bình thường gần đây (ngưỡng thích ứng: camera pan nhanh cũng làm histogram đổi);
  - chuyển cảnh mờ (crossfade/wipe): so frame hiện tại với frame `gradual_window` frame trước.
Sau mỗi lần cắt, bỏ qua `min_shot_len` frame để một crossfade không bị đếm thành nhiều cut.
"""
from collections import deque

import cv2
import numpy as np


class ShotDetector:
    def __init__(self, threshold=0.4, adaptive=3.0, gradual_threshold=0.5, gradual_window=8,
                 min_shot_len=10, history=30):
        self.threshold, self.adaptive = threshold, adaptive
        self.gradual_threshold, self.gradual_window = gradual_threshold, gradual_window
        self.min_shot_len = min_shot_len
        self.diffs = deque(maxlen=history)               # khoảng cách giữa các frame liền kề gần đây
        self.hists = deque(maxlen=gradual_window + 1)
        self.shot_id = -1
        self.since_cut = 0
        self.last_diff = 0.0

    @staticmethod
    def _hist(frame):
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        return cv2.normalize(h, h, 1, 0, cv2.NORM_L1)

    def update(self, frame) -> bool:
        """True nếu frame này mở đầu một shot mới (frame đầu tiên cũng tính là cut)."""
        h = self._hist(frame)
        if not self.hists:
            cut = True
            self.last_diff = 0.0
        else:
            d = cv2.compareHist(self.hists[-1], h, cv2.HISTCMP_BHATTACHARYYA)
            self.last_diff = d
            typical = float(np.median(self.diffs)) if self.diffs else 0.0
            hard = d > max(self.threshold, self.adaptive * typical)
            gradual = (len(self.hists) > self.gradual_window and
                       cv2.compareHist(self.hists[0], h, cv2.HISTCMP_BHATTACHARYYA) > self.gradual_threshold)
            cut = (hard or gradual) and self.since_cut >= self.min_shot_len
            self.diffs.append(d)
        if cut:
            self.shot_id += 1
            self.since_cut = 0
            self.hists.clear()
        self.since_cut += 1
        self.hists.append(h)
        return cut
