"""Detect cầu thủ nhỏ (ở xa) bằng cắt tile, kiểu SAHI.

Full frame chạy ở imgsz cố định -> cầu thủ xa chỉ còn vài chục pixel, detector dễ bỏ sót. Cắt vùng xa camera
(mặc định nửa trên frame, camera chính nhìn từ khán đài) thành lưới tile chồng nhau, phóng to từng tile lên
TILE_IMGSZ rồi detect, đổi toạ độ về frame gốc và gộp với kết quả full frame:
  - box full frame giữ nguyên;
  - box từ tile chỉ thêm vào nếu không trùng box đã có (IoS = giao / diện tích box nhỏ hơn < TILE_IOS):
    dùng IoS thay IoU vì cầu thủ bị mép tile cắt đôi chỉ còn nửa box -> IoU thấp nhưng IoS cao.
"""
import numpy as np


def tile_boxes(w, h, grid, region, overlap):
    """Toạ độ các tile (x1, y1, x2, y2) phủ vùng region (tỉ lệ x1, y1, x2, y2) bằng lưới grid (cột, hàng)."""
    rx1, ry1, rx2, ry2 = int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h)
    cols, rows = grid
    tw = (rx2 - rx1) / (cols - (cols - 1) * overlap)
    th = (ry2 - ry1) / (rows - (rows - 1) * overlap)
    out = []
    for r in range(rows):
        for c in range(cols):
            x1 = rx1 + c * tw * (1 - overlap)
            y1 = ry1 + r * th * (1 - overlap)
            out.append((int(x1), int(y1), int(min(x1 + tw, rx2)), int(min(y1 + th, ry2))))
    return out


def ios(b, bs):
    """Giao / diện tích box nhỏ hơn, giữa box b và các box bs."""
    if not len(bs):
        return np.zeros(0)
    x1 = np.maximum(b[0], bs[:, 0]); y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2]); y2 = np.minimum(b[3], bs[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1]); ab = (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])
    return inter / np.maximum(np.minimum(a, ab), 1e-9)


def tiled_detect(model, frame, imgsz, conf, classes, device, grid=(3, 1), overlap=0.2, region=(0, 0, 1, 0.6),
                 full=None, ios_th=0.5, max_h=None, upscale=1.5):
    """-> (boxes xyxy (N,4), conf (N,), cls (N,), số box thêm từ tile).

    imgsz: None -> cạnh dài tile x upscale (làm tròn bội 32), tức tile được phóng to `upscale` lần so với ảnh gốc.

    full: (boxes, conf, cls) đã detect trên full frame -> gộp; None -> chỉ trả về box từ tile.
    max_h: chỉ lấy box từ tile thấp hơn N px (tile chỉ để bắt cầu thủ nhỏ; box lớn full frame đã thấy)."""
    h, w = frame.shape[:2]
    tiles = tile_boxes(w, h, grid, region, overlap)
    crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
    if imgsz is None:
        side = max(max(x2 - x1, y2 - y1) for x1, y1, x2, y2 in tiles)
        imgsz = int(round(side * upscale / 32)) * 32
    res = model.predict(crops, imgsz=imgsz, conf=conf, classes=classes, device=device, verbose=False)
    tb, tc, tk = [], [], []
    for (x1, y1, x2, y2), r in zip(tiles, res):
        if r.boxes is None or not len(r.boxes):
            continue
        b = r.boxes.xyxy.cpu().numpy() + [x1, y1, x1, y1]
        c = r.boxes.conf.cpu().numpy()
        k = r.boxes.cls.cpu().numpy()
        # bỏ box chạm mép tile ở phía tiếp giáp tile khác (cầu thủ bị cắt) — tile bên cạnh / full frame sẽ thấy đủ
        inner = np.ones(len(b), bool)
        m = 2
        if x1 > 0:
            inner &= b[:, 0] > x1 + m
        if x2 < w:
            inner &= b[:, 2] < x2 - m
        if y1 > 0:
            inner &= b[:, 1] > y1 + m
        if y2 < h:
            inner &= b[:, 3] < y2 - m
        if max_h:
            inner &= (b[:, 3] - b[:, 1]) < max_h
        tb.append(b[inner]); tc.append(c[inner]); tk.append(k[inner])
    tb = np.concatenate(tb) if tb else np.zeros((0, 4))
    tc = np.concatenate(tc) if tc else np.zeros(0)
    tk = np.concatenate(tk) if tk else np.zeros(0)

    kb, kc, kk = (list(full[0]), list(full[1]), list(full[2])) if full is not None else ([], [], [])
    n_full = len(kb)
    for i in np.argsort(-tc):                              # tile: conf cao trước, bỏ box trùng box đã giữ
        if len(kb) and ios(tb[i], np.asarray(kb)).max() >= ios_th:
            continue
        kb.append(tb[i]); kc.append(tc[i]); kk.append(tk[i])
    boxes = np.asarray(kb, dtype=np.float32).reshape(-1, 4)
    return boxes, np.asarray(kc, dtype=np.float32), np.asarray(kk, dtype=np.float32), len(kb) - n_full


class OwnTracker:
    """Tracker của ultralytics (ByteTrack / BoT-SORT) gọi trực tiếp, để đưa vào detection đã gộp (full + tile).
    update() -> mảng [x1, y1, x2, y2, track_id, score, cls, idx] như model.track()."""

    def __init__(self, tracker_yaml):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
        if getattr(cfg, "with_reid", False):
            raise SystemExit("--tile: tracker có with_reid=True chưa được hỗ trợ (cần feature của detector)")
        self.tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)

    def update(self, boxes, conf, cls, frame):
        from ultralytics.engine.results import Boxes

        data = np.c_[boxes, conf, cls].astype(np.float32) if len(boxes) else np.zeros((0, 6), np.float32)
        out = self.tracker.update(Boxes(data, frame.shape[:2]), frame)
        if len(out):          # box từ Kalman có thể tràn khỏi khung hình (model.track() tự clip, ở đây phải tự làm)
            h, w = frame.shape[:2]
            out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
            out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
        return out

    def reset(self):
        self.tracker.reset()
