"""Minimap 2D: vẽ sân, chấm cầu thủ (kèm số áo), vùng camera đang nhìn; làm mượt vị trí offline."""
from collections import defaultdict

import cv2
import numpy as np

from pipeline.pitch import PitchConfig, pitch_lines, project
from pipeline.team import text_ink

# BGR. Chưa chia đội (hoặc track chưa rõ đội) -> màu theo vai trò; có đội -> màu áo của đội (palette).
COLORS = {
    "player": (255, 255, 255),        # trắng, chữ đen
    "player_unknown": (190, 190, 190),
    "goalkeeper": (0, 215, 255),      # vàng
    "referee": (20, 20, 20),          # đen, chữ trắng
}
GRASS = (40, 110, 40)
LINE = (235, 235, 235)


class MinimapRenderer:
    def __init__(self, cfg: PitchConfig, width=420, margin_m=3.0, palette=None):
        self.cfg = cfg
        self.palette = palette or {}                                    # đội -> BGR
        self.margin = margin_m
        self.scale = width / (cfg.length + 2 * margin_m)              # px / mét
        self.size = (width, int(round((cfg.width + 2 * margin_m) * self.scale)))
        self.radius = max(6, int(round(2.0 * self.scale)))
        self.base = self._draw_pitch()

    def to_px(self, xy):
        p = (np.asarray(xy, dtype=np.float64).reshape(-1, 2) + self.margin) * self.scale
        return np.round(p).astype(np.int32)

    def _draw_pitch(self):
        w, h = self.size
        img = np.full((h, w, 3), GRASS, np.uint8)
        lines, spots = pitch_lines(self.cfg)
        for ln in lines:
            cv2.polylines(img, [self.to_px(ln)], False, LINE, 1, cv2.LINE_AA)
        for p in self.to_px(spots):
            cv2.circle(img, tuple(int(v) for v in p), 2, LINE, -1, cv2.LINE_AA)
        return img

    def view_polygon(self, H, shape):
        """Vùng sân camera đang nhìn thấy (mét). Cạnh trên frame thường ở trên chân trời -> chỉ lấy
        phần frame có w > 0, cắt ở 40% chiều cao nếu cần."""
        h, w = shape[:2]
        for top in (0.0, 0.2, 0.4):
            quad = [(0, top * h), (w - 1, top * h), (w - 1, h - 1), (0, h - 1)]
            q, ww = project(H, quad)
            if np.all(ww > 0) and np.all(np.isfinite(q)) and np.abs(q).max() < 1e4:
                return q
        return None

    def draw(self, dots, view=None, stale=False, label=None):
        """dots: [(x_m, y_m, role, number[, team])]. role: player | goalkeeper | referee. number <= 0 = chưa biết.
        team: 0 | 1 | -1 (chưa rõ). Thủ môn có đội: nền vàng, viền màu đội.
        stale=True: frame hiện tại không có phép chiếu -> vẽ mờ vị trí cuối cùng đã biết."""
        img = self.base.copy()
        if view is not None:
            over = img.copy()
            cv2.fillPoly(over, [self.to_px(view)], (70, 150, 70))
            img = cv2.addWeighted(over, 0.5, img, 0.5, 0)
            cv2.polylines(img, [self.to_px(view)], True, (120, 200, 120), 1, cv2.LINE_AA)
        r = self.radius
        for x, y, role, number, *rest in dots:
            team = rest[0] if rest else -1
            px, py = (int(v) for v in self.to_px([(x, y)])[0])
            team_color = self.palette.get(team)
            if team_color is not None and role == "player":
                color = team_color
            else:
                key = "player_unknown" if role == "player" and number <= 0 else role
                color = COLORS.get(key, COLORS["player"])
            cv2.circle(img, (px, py), r, color, -1, cv2.LINE_AA)
            if team_color is not None and role == "goalkeeper":
                cv2.circle(img, (px, py), r, team_color, 2, cv2.LINE_AA)
            cv2.circle(img, (px, py), r + (1 if role == "goalkeeper" else 0), (0, 0, 0), 1, cv2.LINE_AA)
            if number > 0:
                text = str(number)
                fs = 0.32 if len(text) == 1 else 0.28
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                ink = text_ink(color)
                cv2.putText(img, text, (px - tw // 2, py + th // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, ink, 1,
                            cv2.LINE_AA)
        if stale:
            img = cv2.addWeighted(img, 0.45, np.zeros_like(img), 0.55, 0)
        if label:
            cv2.putText(img, label, (6, self.size[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return img


def overlay(frame, mini, pos="bottom-right", pad=12, alpha=0.9):
    """Chèn minimap vào góc frame (in-place)."""
    H, W = frame.shape[:2]
    h, w = mini.shape[:2]
    if h + pad > H or w + pad > W:
        return frame
    y = H - h - pad if pos.startswith("bottom") else pad
    x = {"left": pad, "center": (W - w) // 2, "right": W - w - pad}[pos.split("-")[1]]
    roi = frame[y:y + h, x:x + w]
    frame[y:y + h, x:x + w] = cv2.addWeighted(mini, alpha, roi, 1 - alpha, 0)
    return frame


def _densify(line, step=1.0):
    """Chia polyline thành đoạn <= step mét (để cắt đúng phần nằm sau chân trời khi chiếu ngược)."""
    out = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * t for t in np.arange(1, k + 1) / k)
    return np.array(out)


def draw_debug(frame, cfg: PitchConfig, H, status, dbg, kpt_conf):
    """--pitch-debug: vẽ vạch sân chiếu ngược H⁻¹ lên frame (phải trùng vạch thật) + keypoints + trạng thái."""
    if H is not None:
        Hinv = np.linalg.inv(H)
        lines, _ = pitch_lines(cfg)
        for ln in lines:
            p, w = project(Hinv, _densify(ln))
            ok = (w > 0) & np.isfinite(p).all(1) & (np.abs(p) < 1e5).all(1)
            for a, b, oa, ob in zip(p[:-1], p[1:], ok[:-1], ok[1:]):
                if oa and ob:
                    cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 0, 255), 2, cv2.LINE_AA)
    dbg = dbg or {}
    if "kp_xy" in dbg:
        for i, ((x, y), c) in enumerate(zip(dbg["kp_xy"], dbg["kp_conf"])):
            if x <= 1 and y <= 1:
                continue
            color = (0, 255, 0) if c >= kpt_conf else (0, 0, 255)
            cv2.circle(frame, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
            cv2.putText(frame, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
    text = f"shot {dbg.get('shot', '?')} | H: {status}"
    if "n_kp" in dbg:
        text += f" | kp {dbg['n_kp']}"
    if "n_inliers" in dbg:
        text += f" inl {dbg['n_inliers']}"
    if "n_lines" in dbg:
        text += f" lines {dbg['n_lines']}"
    if "err_m" in dbg:
        text += f" err {dbg['err_m']:.2f}m"
    if "rep_err_px" in dbg:
        text += f" err {dbg['rep_err_px']:.1f}px"
    if "rejected" in dbg:
        text += f" (rejected: {dbg['rejected']})"          # cv2.putText không vẽ được chữ có dấu
    cv2.rectangle(frame, (0, 0), (12 + 9 * len(text), 30), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def _runs(frames):
    """Tách dãy frame tăng dần thành các đoạn liên tiếp (chỉ số)."""
    cut = np.where(np.diff(frames) > 1)[0] + 1
    return np.split(np.arange(len(frames)), cut)


def _filter(v, k, fn):
    if len(v) < 3:
        return v
    k = min(k, len(v) if len(v) % 2 else len(v) - 1)
    p = np.pad(v, k // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(p, k)
    return fn(win, axis=1)


def smooth_positions(rows, max_gap=5, median_k=5, mean_k=5):
    """rows: [(frame, key, x | None, y | None)], key = (track_id, shot_id).
    Mỗi (track, shot): nội suy lỗ hổng <= max_gap frame (frame có detect nhưng không có phép chiếu),
    rồi median (bỏ nhảy cóc) + trung bình trượt (bỏ rung). Trả {(frame, track_id): (x, y)}."""
    by_key = defaultdict(list)
    for f, key, x, y in rows:
        by_key[key].append((f, np.nan if x is None else x, np.nan if y is None else y))
    out = {}
    for (tid, _), seq in by_key.items():
        seq.sort()
        a = np.array(seq, dtype=np.float64)
        frames = a[:, 0].astype(int)
        for run in _runs(frames):
            xy = a[run, 1:]
            ok = np.isfinite(xy[:, 0])
            if not ok.any():
                continue
            # nội suy các lỗ hổng ngắn ở giữa; lỗ dài hoặc ở 2 đầu thì để trống
            idx = np.arange(len(run))
            fill = ~ok
            if fill.any():
                gap_id = np.cumsum(ok)                    # lỗ hổng giữa 2 điểm hợp lệ có cùng gap_id
                for g in np.unique(gap_id[fill]):
                    m = fill & (gap_id == g)
                    left_ok = g > 0
                    right_ok = (idx > idx[m].max()).any() and ok[idx > idx[m].max()].any()
                    if left_ok and right_ok and m.sum() <= max_gap:
                        for c in range(2):
                            xy[m, c] = np.interp(idx[m], idx[ok], xy[ok, c])
            ok = np.isfinite(xy[:, 0])
            for sub in _runs(idx[ok]):                     # làm mượt từng đoạn liền mạch
                j = idx[ok][sub]
                sx = _filter(_filter(xy[j, 0], median_k, np.median), mean_k, np.mean)
                sy = _filter(_filter(xy[j, 1], median_k, np.median), mean_k, np.mean)
                for jj, x, y in zip(j, sx, sy):
                    out[(int(frames[run[jj]]), tid)] = (float(x), float(y))
    return out
