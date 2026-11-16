"""Cắt vùng thân áo (torso) từ keypoints COCO — giống helpers.get_points/generate_crops."""
from typing import Optional

import numpy as np

# COCO keypoints: 5 vai trái, 6 vai phải, 11 hông trái, 12 hông phải
TORSO_KPTS = [6, 5, 11, 12]


def torso_box(kpts: np.ndarray, frame_shape, kpt_conf: float, padding: int) -> Optional[tuple]:
    """kpts: (17, 3) [x, y, conf] theo toạ độ frame. Trả về (x1, y1, x2, y2) hoặc None."""
    if kpts is None or len(kpts) < 13:
        return None
    pts = kpts[TORSO_KPTS]
    if np.any(pts[:, 2] < kpt_conf):
        return None
    h, w = frame_shape[:2]
    x1 = max(0, int(pts[:, 0].min() - padding))
    x2 = min(w - 1, int(pts[:, 0].max() + padding))
    y1 = max(0, int(pts[:, 1].min() - padding))
    y2 = min(h - 1, int(pts[:, 1].max()))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def torso_from_bbox(box, frame_shape, ratio) -> Optional[tuple]:
    """Cắt thân áo theo tỉ lệ cố định của bbox, không cần pose.
    ratio = (x1, y1, x2, y2) theo tỉ lệ chiều rộng/cao của box."""
    h_img, w_img = frame_shape[:2]
    bx1, by1, bx2, by2 = [float(v) for v in box[:4]]
    w, h = bx2 - bx1, by2 - by1
    x1 = max(0, int(bx1 + ratio[0] * w))
    y1 = max(0, int(by1 + ratio[1] * h))
    x2 = min(w_img - 1, int(bx1 + ratio[2] * w))
    y2 = min(h_img - 1, int(by1 + ratio[3] * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def match_pose_to_box(box, pose_boxes: np.ndarray) -> int:
    """Chọn pose có IoU lớn nhất với box cầu thủ (dùng khi chạy pose model riêng). -1 nếu không có."""
    if len(pose_boxes) == 0:
        return -1
    x1 = np.maximum(box[0], pose_boxes[:, 0])
    y1 = np.maximum(box[1], pose_boxes[:, 1])
    x2 = np.minimum(box[2], pose_boxes[:, 2])
    y2 = np.minimum(box[3], pose_boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    iou = inter / (area(np.asarray(box)) + area(pose_boxes) - inter + 1e-6)
    i = int(iou.argmax())
    return i if iou[i] > 0.5 else -1
