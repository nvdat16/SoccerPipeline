"""Camera đang nhìn vào đâu trên sân: pitch keypoints -> homography ảnh (pixel) -> sân (mét).

Mỗi frame:
  1. Model YOLO-pose (Roboflow `sports`, football-pitch-detection.pt) tìm 32 điểm mốc trên sân.
  2. Điểm đủ tin cậy + toạ độ thật của chúng trên sân -> cv2.findHomography (RANSAC).
  3. Kiểm tra H có hợp lý không (lỗi reprojection, sân nhìn thấy có diện tích hợp lý, không nhảy).
  4. Camera chuyển động giữa 2 frame (optical flow) -> dự đoán H từ frame trước. Dùng để
     (a) lấp frame thiếu điểm mốc trong cùng shot, (b) làm mượt H mà không bị trễ khi camera pan.
  Cắt cảnh -> reset() (góc máy mới, H cũ vô nghĩa).
  5. Sau cả video, fill_gaps(): frame vẫn chưa có H (đầu shot trước lần đo đầu tiên, hoặc H mất giữa shot
     tới lần đo tốt tiếp theo) -> lan NGƯỢC từ frame có H phía sau qua chuyển động camera đã lưu (bước 4 chỉ
     lan xuôi được).

Hệ toạ độ sân: gốc (0, 0) = góc sân trái-trên của minimap, x dọc chiều dài (0..L), y dọc chiều rộng
(0..W), đơn vị mét.
"""
import time
from dataclasses import dataclass
from math import sqrt

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchConfig:
    """Kích thước sân (mét). Mặc định theo luật FIFA, sân 105 x 68.

    32 điểm mốc giữ ĐÚNG THỨ TỰ keypoint của model Roboflow (SoccerPitchConfiguration trong repo
    roboflow/sports). Nhưng toạ độ dùng kích thước thật: config gốc của Roboflow là sân 120 x 70 với
    vòng cấm sâu 20.15 m (không theo luật), trong khi model nhận ra các vạch THẬT trên sân -> dùng
    toạ độ Roboflow thì homography bị méo.
    """
    length: float = 105.0
    width: float = 68.0
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    @property
    def arc_half_chord(self) -> float:
        """Nửa dây cung: cung phạt đền cắt vạch vòng cấm cách trục giữa sân bao nhiêu mét."""
        d = self.penalty_box_length - self.penalty_spot_distance
        return sqrt(max(self.centre_circle_radius ** 2 - d ** 2, 0.0))

    @property
    def vertices(self) -> np.ndarray:
        """(32, 2) toạ độ sân của 32 keypoint, theo thứ tự output của model."""
        L, W = self.length, self.width
        pb, gb = self.penalty_box_width / 2, self.goal_box_width / 2
        PL, GL, PS, R = self.penalty_box_length, self.goal_box_length, self.penalty_spot_distance, \
            self.centre_circle_radius
        cy = W / 2
        # keypoint 10, 11 (0-based) nằm trên vạch vòng cấm, giữa 2 góc vòng cấm. Roboflow đặt chúng
        # ngang hàng vạch vòng 5m50 (±9.16 m) -- với sân Roboflow đó cũng đúng là chỗ cung phạt đền
        # chạm vạch vòng cấm. Trên sân thật, cung phạt đền cắt vạch vòng cấm ở ±arc_half_chord (7.31 m).
        a = self.arc_half_chord
        v = [
            (0, 0), (0, cy - pb), (0, cy - gb), (0, cy + gb), (0, cy + pb), (0, W),     # 0-5 vạch cầu môn trái
            (GL, cy - gb), (GL, cy + gb),                                              # 6-7 vòng 5m50 trái
            (PS, cy),                                                                  # 8 chấm 11m trái
            (PL, cy - pb), (PL, cy - a), (PL, cy + a), (PL, cy + pb),                  # 9-12 vạch vòng cấm trái
            (L / 2, 0), (L / 2, cy - R), (L / 2, cy + R), (L / 2, W),                  # 13-16 vạch giữa sân
            (L - PL, cy - pb), (L - PL, cy - a), (L - PL, cy + a), (L - PL, cy + pb),  # 17-20 vòng cấm phải
            (L - PS, cy),                                                              # 21 chấm 11m phải
            (L - GL, cy - gb), (L - GL, cy + gb),                                      # 22-23 vòng 5m50 phải
            (L, 0), (L, cy - pb), (L, cy - gb), (L, cy + gb), (L, cy + pb), (L, W),    # 24-29 vạch cầu môn phải
            (L / 2 - R, cy), (L / 2 + R, cy),                                          # 30-31 mép vòng tròn giữa
        ]
        return np.asarray(v, dtype=np.float32)


def pitch_lines(cfg: PitchConfig):
    """Các vạch sân dạng polyline (mét) + các chấm (tâm, chấm 11m). Dùng cho minimap và debug."""
    L, W, cy = cfg.length, cfg.width, cfg.width / 2
    pb, gb = cfg.penalty_box_width / 2, cfg.goal_box_width / 2
    R = cfg.centre_circle_radius
    lines = [np.array([(0, 0), (L, 0), (L, W), (0, W), (0, 0)]),
             np.array([(L / 2, 0), (L / 2, W)])]
    t = np.linspace(0, 2 * np.pi, 73)
    lines.append(np.stack([L / 2 + R * np.cos(t), cy + R * np.sin(t)], 1))
    for x0, s in ((0.0, 1.0), (L, -1.0)):           # s = hướng vào trong sân
        for depth, half in ((cfg.penalty_box_length, pb), (cfg.goal_box_length, gb)):
            lines.append(np.array([(x0, cy - half), (x0 + s * depth, cy - half),
                                   (x0 + s * depth, cy + half), (x0, cy + half)]))
        # cung phạt đền: phần đường tròn tâm chấm 11m nằm ngoài vòng cấm
        d = cfg.penalty_box_length - cfg.penalty_spot_distance
        th = np.arccos(np.clip(d / R, -1, 1))
        t = np.linspace(-th, th, 25)
        lines.append(np.stack([x0 + s * (cfg.penalty_spot_distance + R * np.cos(t)), cy + R * np.sin(t)], 1))
    spots = np.array([(L / 2, cy), (cfg.penalty_spot_distance, cy), (L - cfg.penalty_spot_distance, cy)])
    return [ln.astype(np.float64) for ln in lines], spots


def project(H: np.ndarray, pts) -> tuple:
    """Áp homography lên (N, 2) điểm. Trả (xy (N, 2), w (N,)); w <= 0 = điểm nằm sau đường chân trời."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    w = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[:, :2] / w[:, None]
    return xy, w


def _view_rows(shape, top=0.4):
    """Lưới 3x3 điểm ảnh ở phần dưới frame (từ `top` * h xuống đáy) -- vùng gần như luôn là mặt sân."""
    h, w = shape[:2]
    xs = np.array([0.0, 0.5, 1.0]) * (w - 1)
    ys = np.array([top, (1 + top) / 2, 1.0]) * (h - 1)
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float64)


class PitchCalibrator:
    FLOW_WIDTH = 640           # optical flow chạy trên frame thu nhỏ
    FLOW_IGNORE_TOP = 0.2      # bỏ 20% trên cùng (bảng tỉ số, logo kênh đứng yên -> kéo camera motion về 0)

    def __init__(self, model_path, device, cfg: PitchConfig, imgsz=640, det_conf=0.3, kpt_conf=0.5,
                 ransac_m=1.0, max_reproj_m=1.0, min_inliers=4, max_jump_m=5.0, ema=0.5,
                 max_flow_frames=50, switch_after=12, pnl=None, max_reproj_px=8.0):
        """pnl: PnLCalibModel (pipeline/calib_pnl.py) -> dùng nó thay cho model 32 điểm Roboflow."""
        from ultralytics import YOLO
        self.pnl, self.max_reproj_px = pnl, max_reproj_px
        self.model = YOLO(str(model_path)) if model_path is not None and pnl is None else None
        self.device, self.cfg, self.imgsz = device, cfg, imgsz
        self.det_conf, self.kpt_conf = det_conf, kpt_conf
        self.ransac_m, self.max_reproj_m, self.min_inliers = ransac_m, max_reproj_m, min_inliers
        self.max_jump_m, self.ema, self.max_flow_frames = max_jump_m, ema, max_flow_frames
        self.switch_after = switch_after
        self.V = cfg.vertices.astype(np.float64)
        self.reset()

    def reset(self):
        self.H = None              # H (ảnh -> sân) của frame trước, None = chưa/không biết
        self.prev_gray = None
        self.flow_frames = 0       # số frame liên tiếp chỉ dựa vào optical flow
        self.cand = None           # H mới bị loại vì nhảy xa (ứng viên thay H hiện tại)
        self.cand_n = 0            # số lần đo liên tiếp khớp với ứng viên

    # ------------------------------------------------------------------ keypoints -> H
    def measure(self, frame):
        """Một lần đo homography từ chính frame này (chưa xét liên tục theo thời gian): (H | None, info)."""
        if self.pnl is not None:
            H, info = self.pnl.estimate(frame)
            if H is not None:
                H = self._normalise(H, frame.shape)
                if H is None or info.get("rep_err_px", np.inf) > self.max_reproj_px:
                    H = None
                elif not self.plausible(H, frame.shape):
                    H, info["rejected"] = None, "implausible"
            return H, info
        t0 = time.perf_counter()
        xy, conf = self.keypoints(frame)
        info = {"model_ms": (time.perf_counter() - t0) * 1000}
        if xy is None:
            return None, info
        info["kp_xy"], info["kp_conf"] = xy, conf
        H, fit_info = self.fit(xy, conf, frame.shape)
        info.update(fit_info)
        return H, info

    def keypoints(self, frame):
        """(xy (32, 2), conf (32,)) của sân nhận ra tốt nhất, hoặc (None, None)."""
        r = self.model(frame, imgsz=self.imgsz, conf=self.det_conf, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None, None
        i = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        xy = r.keypoints.xy[i].cpu().numpy().astype(np.float64)
        conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        return xy, conf

    def fit(self, xy, conf, shape):
        """Homography từ keypoints, hoặc None nếu không đủ tin cậy. Trả (H, info)."""
        ok = (conf >= self.kpt_conf) & (xy[:, 0] > 1) & (xy[:, 1] > 1)   # ultralytics trả (0,0) cho điểm ẩn
        info = {"n_kp": int(ok.sum())}
        if ok.sum() < 4:
            return None, info
        src, dst = xy[ok], self.V[ok]
        if cv2.contourArea(cv2.convexHull(dst.astype(np.float32))) < 30:   # các điểm gần thẳng hàng
            return None, info
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_m)
        if H is None:
            return None, info
        inl = inl.ravel().astype(bool)
        info["n_inliers"] = int(inl.sum())
        if inl.sum() < self.min_inliers:
            return None, info
        p, _ = project(H, src[inl])
        info["err_m"] = float(np.median(np.linalg.norm(p - dst[inl], axis=1)))
        if info["err_m"] > self.max_reproj_m:
            return None, info
        H = self._normalise(H, shape)
        if H is None or not self.plausible(H, shape):
            return None, info
        return H, info

    @staticmethod
    def _normalise(H, shape):
        """Đổi dấu sao cho w > 0 ở đáy frame; chia cho H[2,2]."""
        h, w = shape[:2]
        _, ww = project(H, [(w / 2, h - 1)])
        if not np.isfinite(ww[0]) or abs(ww[0]) < 1e-12:
            return None
        return H if ww[0] > 0 else -H

    def plausible(self, H, shape) -> bool:
        """Phần dưới frame chiếu lên sân phải là tứ giác thật, kích thước hợp lý, nằm quanh sân."""
        L, W = self.cfg.length, self.cfg.width
        h, w = shape[:2]
        quad = np.array([(0, 0.4 * h), (w - 1, 0.4 * h), (w - 1, h - 1), (0, h - 1)])
        q, ww = project(H, quad)
        if not np.all(ww > 0) or not np.all(np.isfinite(q)):
            return False                                   # có góc nằm sau chân trời -> không phải mặt sân
        area = cv2.contourArea(q.astype(np.float32))
        if not 30 < area < 2 * L * W:                      # close-up (vài m²) hoặc phép chiếu bùng nổ
            return False
        bc, _ = project(H, [(w / 2, h - 1)])
        x, y = bc[0]
        return -20 < x < L + 20 and -20 < y < W + 20

    def _jump(self, Ha, Hb, shape) -> float:
        g = _view_rows(shape)
        a, _ = project(Ha, g)
        b, _ = project(Hb, g)
        return float(np.median(np.linalg.norm(a - b, axis=1)))

    def _blend(self, H_meas, H_ref, shape):
        """Trộn H đo được với H dự đoán (EMA) trên lưới điểm ảnh -> không làm méo phép chiếu."""
        g = _view_rows(shape)
        a, _ = project(H_meas, g)
        b, _ = project(H_ref, g)
        H, _ = cv2.findHomography(g, self.ema * a + (1 - self.ema) * b, 0)
        return self._normalise(H, shape) if H is not None else H_meas

    # ------------------------------------------------------------------ camera motion
    def _gray(self, frame):
        s = self.FLOW_WIDTH / frame.shape[1]
        return cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY), s

    def _camera_motion(self, prev, cur, boxes, s):
        """Homography M: ảnh frame trước -> ảnh frame này (full-res), hoặc None nếu không ước lượng được."""
        mask = np.full(prev.shape, 255, np.uint8)
        mask[: int(self.FLOW_IGNORE_TOP * prev.shape[0])] = 0
        for x1, y1, x2, y2 in boxes:                       # cầu thủ tự chuyển động -> không dùng
            pad = 0.2 * (x2 - x1)
            mask[max(int((y1 - pad) * s), 0): int((y2 + pad) * s) + 1,
                 max(int((x1 - pad) * s), 0): int((x2 + pad) * s) + 1] = 0
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=mask)
        if p0 is None or len(p0) < 30:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
        st = st.ravel().astype(bool)
        if st.sum() < 30:
            return None
        M, inl = cv2.findHomography(p0[st], p1[st], cv2.RANSAC, 1.5)
        if M is None or inl.sum() < max(20, 0.5 * st.sum()):
            return None
        S = np.diag([s, s, 1.0])
        return np.linalg.inv(S) @ M @ S

    # ------------------------------------------------------------------ main
    def update(self, frame, is_cut: bool, boxes=(), run_model: bool = True):
        """Trả (H | None, status, info). status: 'kp' (từ keypoints), 'flow' (nối bằng optical flow), 'none'."""
        if is_cut:
            self.reset()
        gray, s = self._gray(frame)
        info = {}

        pred = None                                        # H của frame trước, dời theo chuyển động camera
        M = None                                           # luôn tính (kể cả khi chưa có H) -> fill_gaps lan ngược
        if self.prev_gray is not None:
            M = self._camera_motion(self.prev_gray, gray, boxes, s)
            if M is not None and self.H is not None:
                pred = self._normalise(self.H @ np.linalg.inv(M), frame.shape)
        info["motion"] = M
        ref = pred if pred is not None else self.H

        meas, switch = None, False
        if run_model:
            meas, m_info = self.measure(frame)
            info.update(m_info)
            # Trong 1 shot camera không "dịch chuyển tức thời": H mới lệch xa H dự đoán gần như luôn là model
            # sai (hay gặp: nhầm khung thành trái/phải). Chỉ chuyển khi `switch_after` lần đo liên tiếp cùng
            # chỉ về vị trí mới (lúc đó H cũ mới là sai), và chuyển hẳn -- trộn 2 H khác xa nhau ra H rác.
            if meas is not None and ref is not None and self._jump(meas, ref, frame.shape) > self.max_jump_m:
                if self.cand is not None and self._jump(meas, self.cand, frame.shape) <= self.max_jump_m:
                    self.cand_n += 1
                else:
                    self.cand_n = 1
                self.cand = meas
                if self.cand_n < self.switch_after:
                    meas = None
                    info["rejected"] = "jump"
                else:
                    switch = True
            elif meas is not None:
                self.cand, self.cand_n = None, 0

        if meas is not None:
            self.cand, self.cand_n = None, 0
            self.flow_frames = 0
            H, status = (self._blend(meas, ref, frame.shape) if ref is not None and not switch else meas), "kp"
        elif pred is not None and self.flow_frames < self.max_flow_frames:
            self.flow_frames += 1
            H, status = pred, "flow"
        else:
            H, status = None, "none"
        if H is not None and not self.plausible(H, frame.shape):
            H, status = None, "none"

        self.H = H
        self.prev_gray = gray
        return H, status, info

    def fill_gaps(self, homs, motion, shot_of, shape):
        """Lấp frame chưa có H sau khi xử lý xong video.

        homs: frame -> H | None; motion: frame -> M (ảnh frame-1 -> ảnh frame) | None; shot_of: frame -> shot.
        Mỗi đoạn frame liên tiếp không có H trong cùng shot: lan H từ frame có H ngay trước (xuôi: H·M⁻¹) và ngay
        sau (ngược: H·M), tối đa max_flow_frames bước, dừng khi 1 bước không có chuyển động camera hoặc H không hợp
        lý; frame lấp được từ cả 2 phía lấy phía gần hơn. Trả {frame: H}."""
        out = {}
        frames = sorted(homs)
        i = 0
        while i < len(frames):
            f = frames[i]
            if homs[f] is not None:
                i += 1
                continue
            j = i
            while (j + 1 < len(frames) and homs[frames[j + 1]] is None and frames[j + 1] == frames[j] + 1
                   and shot_of.get(frames[j + 1]) == shot_of.get(f)):
                j += 1
            run, best = frames[i:j + 1], {}
            for direction in (1, -1):
                anchor = run[0] - 1 if direction == 1 else run[-1] + 1
                H = homs.get(anchor)
                if H is None or shot_of.get(anchor) != shot_of.get(f):
                    continue
                for k, g in enumerate(run if direction == 1 else run[::-1]):
                    M = motion.get(g if direction == 1 else g + 1)
                    if M is None or k >= self.max_flow_frames:
                        break
                    H = self._normalise(H @ np.linalg.inv(M) if direction == 1 else H @ M, shape)
                    if H is None or not self.plausible(H, shape):
                        break
                    if g not in best or best[g][0] > k:
                        best[g] = (k, H)
            out.update({g: H for g, (_, H) in best.items()})
            i = j + 1
        return out
