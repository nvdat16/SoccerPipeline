"""Chia đội theo màu áo: histogram HSV trên vùng thân áo -> KMeans 2 cụm -> gán đội THEO FRAME.

Đặc trưng mỗi crop (21 chiều, chuẩn hoá L1 rồi lấy căn bậc 2 = khoảng cách Hellinger):
  - 18 bin hue (mỗi bin 20°) cho pixel có màu (S, V đủ lớn)
  - 3 bin không màu: đen / xám / trắng (áo trắng, đen rất hay gặp)
Pixel cỏ (hue xanh lá) bị bỏ; nếu > 70% crop là "cỏ" thì coi là áo xanh lá, giữ nguyên.

Không cần model: cuối video, KMeans trên mẫu của mọi track 'player' (mỗi track tối đa N mẫu để track dài
không lấn át). Sau đó MỖI MẪU được gán đội (mẫu xa cả 2 tâm -> -1: trọng tài/khán giả bị detect nhầm), làm
mượt theo thời gian trong track, và mỗi frame lấy nhãn của mẫu gần nhất. Gán theo frame chứ không theo
cả track vì ByteTrack hay đổi ID giữa 2 cầu thủ khác đội khi họ chạm nhau.

Thủ môn: track 'goalkeeper' có áo khớp màu 1 đội = cầu thủ bị detector nhận nhầm -> gán theo màu.
Thủ môn thật (áo khác cả 2 đội) để team = -1: đã thử gán theo vị trí (cầu thủ sâu nhất / trọng tâm đội /
đếm cầu thủ gần khung thành) nhưng sai khi 1 đội tấn công liên tục trong clip. Trọng tài không có đội.
"""
from collections import defaultdict

import cv2
import numpy as np

N_HUE = 18
GRASS_HUE = (35, 85)          # OpenCV hue 0..179
FEAT_DIM = N_HUE + 3


def jersey_feature(crop: np.ndarray, s_min=60, v_min=50, v_white=170, grass_max_frac=0.7):
    """crop BGR -> (feature 21 chiều, màu BGR trung bình của pixel áo) hoặc None nếu crop quá nhỏ."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    chroma = (s >= s_min) & (v >= v_min)
    grass = chroma & (h >= GRASS_HUE[0]) & (h <= GRASS_HUE[1])
    keep = ~grass if grass.mean() <= grass_max_frac else np.ones_like(grass)
    if keep.sum() < 16:
        return None
    hk, sk, vk, ck = h[keep], s[keep], v[keep], chroma[keep]
    feat = np.zeros(FEAT_DIM, np.float32)
    # dịch hue 5 đơn vị để màu đỏ (quanh 0/179) nằm gọn trong 1 bin
    np.add.at(feat, ((hk[ck] + 5) % 180) * N_HUE // 180, 1)
    ach_v = vk[~ck]
    feat[N_HUE] = (ach_v < v_min).sum()                                    # đen
    feat[N_HUE + 1] = ((ach_v >= v_min) & (ach_v < v_white)).sum()         # xám
    feat[N_HUE + 2] = (ach_v >= v_white).sum()                             # trắng
    feat = np.sqrt(feat / feat.sum())
    color = crop.reshape(-1, 3)[keep].mean(0)
    return feat, color


class TeamClassifier:
    """Gom mẫu (frame, track, crop) trong lúc phân tích, cuối video gọi fit()."""

    def __init__(self, fit_samples=30, min_samples=3, outlier_mad=4.0, smooth=4, seed=0):
        self.fit_samples = fit_samples        # số mẫu tối đa / track dùng để fit KMeans
        self.min_samples = min_samples        # track ít mẫu hơn -> không dùng để fit (vẫn được gán nhãn)
        self.outlier_mad = outlier_mad
        self.smooth = smooth                  # cửa sổ đa số ±N mẫu trong track
        self.gk_color_match = 0.7             # track 'goalkeeper' có >= 70% mẫu khớp màu đội -> là cầu thủ
        self.as_player = set()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.samples = defaultdict(list)      # track -> [(frame, feat, color)]
        self.label = {}                       # track -> (frames[], labels[]) sau fit

    def add(self, tid: int, frame_idx: int, crop: np.ndarray):
        r = jersey_feature(crop)
        if r is not None:
            self.samples[tid].append((frame_idx, r[0], r[1]))

    # ------------------------------------------------------------------ fit
    def fit(self, roles: dict):
        """roles: track -> player | goalkeeper | referee. Trả info (dict).
        Sau đó dùng team_at(track, frame) và track_summary(track)."""
        players = [t for t, s in self.samples.items()
                   if roles.get(t, "player") == "player" and len(s) >= self.min_samples]
        if len(players) < 2:
            return {"ok": False, "reason": f"chỉ có {len(players)} track cầu thủ đủ mẫu"}

        fit_x = []
        for t in players:
            s = self.samples[t]
            pick = self.rng.choice(len(s), min(len(s), self.fit_samples), replace=False)
            fit_x.extend(s[i][1] for i in pick)
        X = np.stack(fit_x).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        cv2.setRNGSeed(self.seed)
        _, lab, centers = cv2.kmeans(X, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel()
        # đội 0 = cụm đông hơn (đánh số ổn định giữa các lần chạy)
        if (lab == 1).sum() > (lab == 0).sum():
            centers, lab = centers[::-1].copy(), 1 - lab

        # ngưỡng outlier theo từng cụm: median + k * MAD của khoảng cách mẫu -> tâm
        d_fit = np.linalg.norm(X - centers[lab], axis=1)
        limit = []
        for k in (0, 1):
            d = d_fit[lab == k]
            med = np.median(d)
            limit.append(med + self.outlier_mad * 1.4826 * (np.median(np.abs(d - med)) + 1e-6))

        # gán nhãn mọi mẫu của mọi track cầu thủ (kể cả track ngắn), rồi làm mượt trong track.
        # Track 'goalkeeper' mà áo khớp màu 1 đội = cầu thủ bị detector nhận nhầm class -> xử lý như cầu thủ.
        n_out, n_all = 0, 0
        colors = {0: [], 1: []}
        self.as_player = set()
        for t, s in self.samples.items():
            role = roles.get(t, "player")
            if role not in ("player", "goalkeeper"):
                continue
            s.sort(key=lambda r: r[0])
            F = np.stack([r[1] for r in s])
            d = np.linalg.norm(F[:, None, :] - centers[None], axis=2)
            k = d.argmin(1)
            raw = np.where(d[np.arange(len(k)), k] <= np.array(limit)[k], k, -1)
            if role == "goalkeeper":
                if (raw >= 0).mean() < self.gk_color_match:
                    continue                                   # áo khác cả 2 đội -> thủ môn thật
                self.as_player.add(t)
            n_out += int((raw < 0).sum())
            n_all += len(raw)
            sm = self._smooth(raw)
            self.label[t] = (np.array([r[0] for r in s]), sm)
            for (_, _, c), kk in zip(s, sm):
                if kk >= 0:
                    colors[kk].append(c)
        palette = [vivid(np.median(np.stack(colors[k]), 0)) if colors[k] else np.array([200.0] * 3) for k in (0, 1)]

        self.centers, self.limit = centers, limit
        size = [sum(1 for t in self.label if self.track_summary(t)["team"] == k) for k in (0, 1)]
        return {"ok": True, "n_player_tracks": len(players), "pct_outlier_samples": round(100 * n_out / max(n_all, 1), 1),
                "n_goalkeeper_as_player": len(self.as_player),
                "teams": [{"team": k, "n_tracks": size[k], "color_bgr": [int(round(c)) for c in palette[k]],
                           "dominant": describe(centers[k])} for k in (0, 1)]}

    def _smooth(self, raw: np.ndarray) -> np.ndarray:
        """Đa số trong cửa sổ ±smooth mẫu (bỏ qua -1); hoà hoặc toàn -1 -> giữ nhãn gốc."""
        out = raw.copy()
        w = self.smooth
        for i in range(len(raw)):
            win = raw[max(0, i - w): i + w + 1]
            c0, c1 = (win == 0).sum(), (win == 1).sum()
            if c0 != c1:
                out[i] = 0 if c0 > c1 else 1
        return out

    # ------------------------------------------------------------------ query
    def team_at(self, tid: int, frame_idx: int) -> int:
        """Đội của track ở frame này = nhãn mẫu gần nhất theo thời gian. -1 nếu chưa rõ."""
        if tid not in self.label:
            return -1
        frames, labels = self.label[tid]
        if len(frames) == 1:
            return int(labels[0])
        i = int(np.searchsorted(frames, frame_idx))
        if i == len(frames) or (i > 0 and frame_idx - frames[i - 1] <= frames[i] - frame_idx):
            i -= 1
        return int(labels[i])

    def track_summary(self, tid: int) -> dict:
        """team = đội chiếm đa số; team_conf = tỉ lệ; team_switches = số lần đổi đội (dấu hiệu đổi ID)."""
        if tid not in self.label:
            return {"team": -1, "team_conf": 0.0, "team_switches": 0}
        lab = self.label[tid][1]
        known = lab[lab >= 0]
        if not len(known):
            return {"team": -1, "team_conf": 0.0, "team_switches": 0}
        k = int(np.bincount(known, minlength=2).argmax())
        sw = int((np.diff(known) != 0).sum())
        return {"team": k, "team_conf": round(float((known == k).mean()), 3), "team_switches": sw}


def describe(center: np.ndarray) -> str:
    """Tên gần đúng của màu chiếm ưu thế trong tâm cụm (chỉ để in log)."""
    names = ["đỏ", "cam", "cam vàng", "vàng"] + ["xanh lá"] * 5 + ["xanh lơ"] + ["xanh dương"] * 3 \
        + ["tím"] * 2 + ["hồng"] * 2 + ["đỏ"] + [
             "đen", "xám", "trắng"]
    w = center ** 2
    order = np.argsort(-w)[:2]
    return " + ".join(f"{names[i]} {100 * w[i] / w.sum():.0f}%" for i in order)


def vivid(bgr: np.ndarray) -> np.ndarray:
    """Màu trung bình của áo bị tối/xỉn (bóng, da, số áo) -> tăng S, V để dễ nhìn trên video/minimap.
    Áo không màu (trắng/xám/đen) giữ nguyên."""
    h, s, v = cv2.cvtColor(np.uint8([[np.clip(bgr, 0, 255)]]), cv2.COLOR_BGR2HSV)[0, 0].astype(float)
    if s < 60:
        return np.asarray(bgr, float)
    hsv = np.uint8([[[h, min(255, s * 1.5), max(v, 200)]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(float)


def text_ink(bgr) -> tuple:
    """Chữ đen trên nền sáng, chữ trắng trên nền tối."""
    b, g, r = bgr
    return (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else (255, 255, 255)
