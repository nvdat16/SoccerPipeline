"""Định danh cầu thủ trên minimap: nhìn rõ số áo thì ghi nhớ, mất dấu thì nối lại bằng vị trí + vận tốc + hướng.

Track id của ByteTrack sống rất ngắn (che khuất, cắt cảnh, mất phép chiếu), nên số áo đọc được ở một track không
truyền được sang track sau của cùng cầu thủ. Ở đây làm offline sau khi đã có vị trí (mét) đã làm mượt:

  1. Đoạn (segment) = 1 track, tách thêm ở chỗ đội đổi giữa chừng (ByteTrack đổi ID giữa 2 cầu thủ khác đội).
     Mỗi đoạn có số áo của riêng nó nếu đọc rõ (bỏ phiếu như TrackletVoter).
  2. Với đoạn A kết thúc và đoạn B bắt đầu sau đó (cách <= LINK_MAX_GAP_S giây, cùng đội, cùng vai trò):
       vị trí dự đoán = cuối A + vận tốc A × khoảng trống (tối đa LINK_PREDICT_S giây, sau đó coi như đứng yên)
       bán kính cho phép = LINK_R0 + LINK_SPREAD × khoảng trống (cầu thủ có thể đổi hướng/tăng tốc)
       chi phí = khoảng cách / bán kính + LINK_W_DIR × (độ lệch hướng chạy A và B, khi cả hai đang chạy)
                 + LINK_W_VEL × (độ đổi vận tốc / LINK_VMAX)
     Cùng số áo -> giảm chi phí; khác số áo (đều đọc rõ) -> cấm nối.
     Bỏ cặp nếu 1 cầu thủ khác cùng đội (bất kỳ track nào) cũng đứng gần vị trí dự đoán gần bằng B (đám đông).
  3. Ghép tham lam theo chi phí tăng dần, mỗi đoạn có tối đa 1 đoạn trước / 1 đoạn sau. Bỏ qua cặp mập mờ
     (ứng viên thứ hai gần bằng) trừ khi số áo xác nhận. Không nối 2 chuỗi đã có số áo khác nhau.
  4. Chuỗi = 1 cầu thủ; số áo = bỏ phiếu mọi lần đọc trong chuỗi. Định danh cuối = (đội, số áo):
     các chuỗi cùng đội + cùng số ở những lúc khác nhau là cùng 1 cầu thủ.

Gặp nhau (encounter, ENC_SPLIT): 2 cầu thủ CÙNG ĐỘI cách nhau < ENC_R mét là lúc tracker hay hoán ID (cùng màu
áo, box chồng nhau) hoặc làm đứt track. Track được cắt thành  A | E | B  (E = các frame đang gặp nhau):
  - E không tự nối với ai (vị trí lúc chồng nhau không phân biệt được 2 người);
  - A -> B: tracker giữ được ID qua lần gặp -> luôn giữ (chuyển động lúc 2 người chồng nhau nằm trong nhiễu phép
    chiếu, không đủ để bác tracker); chuyển động ủng hộ rõ phương án hoán ID -> vẫn giữ, chỉ ghi vào thống kê
    'keep_suspect' (ENC_MOTION_SUSPECT=True để đánh dấu rủi ro — trên clip_3400 cờ này toàn báo động giả).
    Tracker làm đứt track trong lần gặp -> nối lại phần còn trống bằng Hungarian trên chuyển động trước/sau lần
    gặp (bỏ qua luật đám đông với chính các đồng đội trong lần gặp); rủi ro trừ khi số áo hai bên trùng nhau;
  - E thuộc chuỗi của A (hoặc của B nếu track bắt đầu trong lần gặp), trừ khi A và B của cùng track rơi vào 2 chuỗi
    khác nhau (số áo mâu thuẫn) -> bỏ trống.

Kết quả theo từng detection (frame, track): player_id, số áo, nguồn
  'read'      = đoạn này tự đọc được số;
  'linked'    = số được truyền qua chuỗi nối, đường tới 1 đoạn tự đọc không đi qua lần gặp nào;
  'uncertain' = chỉ có được số nhờ nối qua 1 lần gặp đồng đội KHÔNG phân định rõ (hoặc là đoạn E) -> có thể nhầm.
"""
from collections import defaultdict

import numpy as np

from pipeline.aggregator import TrackletVoter


class _Seg:
    __slots__ = ("idx", "tid", "role", "team", "frames", "pts", "reads", "number", "kind", "enc_end", "enc_start",
                 "partners")

    def __init__(self, idx, tid, role, kind="normal"):
        self.idx, self.tid, self.role = idx, tid, role
        self.team = -1
        self.frames, self.pts, self.reads = [], [], []
        self.number = -1
        self.kind = kind              # 'normal' | 'enc' (đang gặp đồng đội)
        self.enc_end = False          # đoạn kết thúc vì bắt đầu gặp đồng đội
        self.enc_start = False        # đoạn bắt đầu ngay sau 1 lần gặp
        self.partners = set()         # track của các đồng đội trong lần gặp liền kề (đoạn E / A trước E / B sau E)


def _runs(flags, gap, min_len):
    """flags: list bool theo thứ tự frame -> các đoạn [i, j] (chỉ số) True liên tiếp, lấp lỗ <= gap, dài >= min_len."""
    runs, i, n = [], 0, len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = last = i
        while j + 1 < n:
            nxt = next((k for k in range(j + 1, min(n, j + gap + 2)) if flags[k]), None)
            if nxt is None:
                break
            j = last = nxt
        if last - i + 1 >= min_len:
            runs.append((i, last))
        i = last + 1
    return runs


def _motion(pts, fps, k, at_end, vmax):
    """pts [(frame, x, y)] -> (frame, vị trí, vận tốc m/s) ở đầu hoặc cuối đoạn, vận tốc fit tuyến tính trên k điểm."""
    sel = pts[-k:] if at_end else pts[:k]
    a = np.asarray(sel, dtype=np.float64)
    f, xy = a[:, 0], a[:, 1:]
    ref = f[-1] if at_end else f[0]
    p = xy[-1] if at_end else xy[0]
    if len(a) < 3 or f[-1] - f[0] < 3:
        return int(ref), p, np.zeros(2)
    t = (f - f[0]) / fps
    v = np.array([np.polyfit(t, xy[:, c], 1)[0] for c in range(2)])
    s = np.linalg.norm(v)
    if s > vmax:
        v *= vmax / s
    return int(ref), p, v


def _runner_up(costs, c):
    """Chi phí tốt nhất trong danh sách (đã sắp xếp) sau khi bỏ 1 phần tử bằng c."""
    rest = list(costs)
    rest.remove(c)
    return rest[0] if rest else np.inf


class IdentityLinker:
    def __init__(self, fps, cfg, voter_kw):
        self.fps = fps or 25.0
        self.max_gap = cfg.LINK_MAX_GAP_S
        self.predict_s = cfg.LINK_PREDICT_S
        self.r0 = cfg.LINK_R0
        self.spread = cfg.LINK_SPREAD
        self.r_max = cfg.LINK_R_MAX
        self.w_dir = cfg.LINK_W_DIR
        self.moving = cfg.LINK_MOVING_MS
        self.ambig = cfg.LINK_AMBIGUITY
        self.k = cfg.LINK_VEL_POINTS
        self.vmax = cfg.LINK_VMAX
        self.crowd_margin = cfg.LINK_CROWD_MARGIN
        self.w_vel = cfg.LINK_W_VEL
        self.enc_split = getattr(cfg, "ENC_SPLIT", False)
        self.enc_r = getattr(cfg, "ENC_R", 1.5)
        self.enc_min = getattr(cfg, "ENC_MIN_FRAMES", 3)
        self.enc_gap = getattr(cfg, "ENC_GAP_FRAMES", 2)
        self.enc_margin = getattr(cfg, "ENC_SWAP_MARGIN", 0.3)
        self.enc_unmatched = getattr(cfg, "ENC_UNMATCHED", 1.0)
        self.enc_motion_suspect = getattr(cfg, "ENC_MOTION_SUSPECT", False)
        self.solver = getattr(cfg, "LINK_SOLVER", "global")
        self.accept = getattr(cfg, "LINK_ACCEPT", 3.0)
        self.reid_margin = getattr(cfg, "REID_MARGIN", 0.06)
        self.reid_frames = getattr(cfg, "REID_FRAMES", 10)
        self.voter_kw = voter_kw
        self.emb = None                   # (frame, track) -> embedding ngoại hình (L2), từ run(..., emb=)

    # ------------------------------------------------------------------ 1. đoạn
    def _encounters(self, t, frames, positions, by_frame, team_of):
        """Với track t: (cờ 'đang gặp đồng đội' theo từng frame, đồng đội gặp ở frame đó)."""
        flags, partners = [], []
        for f in frames:
            k = team_of.get((f, t), -1)
            me = positions.get((f, t))
            near = set()
            if k >= 0 and me is not None:
                near = {tt for tt, kk, x, y in by_frame[f]
                        if tt != t and kk == k and np.hypot(x - me[0], y - me[1]) < self.enc_r}
            flags.append(bool(near))
            partners.append(near)
        return flags, partners

    def _segments(self, dets, positions, team_of, roles, reads, by_frame):
        by_t = defaultdict(list)
        for f, t in dets:
            by_t[t].append(f)
        segs, where = [], {}
        self.n_enc = 0
        for t, frames in by_t.items():
            role = roles.get(t, "player")
            if role == "referee":
                continue
            frames.sort()
            # frame nào thuộc 1 lần gặp đồng đội (đủ dài, lấp lỗ ngắn) -> chỉ số frame -> id lần gặp
            enc_of, enc_partners = {}, {}
            if self.enc_split:
                flags, partners = self._encounters(t, frames, positions, by_frame, team_of)
                for e, (i, j) in enumerate(_runs(flags, self.enc_gap, self.enc_min)):
                    enc_partners[e] = set().union(*partners[i:j + 1])
                    for q in range(i, j + 1):
                        enc_of[q] = e
                self.n_enc += len(enc_partners)
            cur, cur_team, cur_enc = _Seg(len(segs), t, role), -1, None
            for q, f in enumerate(frames):
                k = team_of.get((f, t), -1)
                e = enc_of.get(q)
                new_seg = None
                if e != cur_enc and cur.frames:               # vào / ra / đổi lần gặp
                    new_seg = _Seg(-1, t, role, kind="enc" if e is not None else "normal")
                    if e is not None:
                        cur.enc_end = cur.kind == "normal"
                        cur.partners |= enc_partners[e]
                        new_seg.partners = set(enc_partners[e])
                    else:
                        new_seg.enc_start = True
                        new_seg.partners = set(cur.partners)
                elif k >= 0 and cur_team >= 0 and k != cur_team and cur.frames:
                    new_seg = _Seg(-1, t, role, kind=cur.kind)   # đổi đội giữa chừng = đổi ID
                if new_seg is not None:
                    cur.team = cur_team
                    segs.append(cur)
                    cur, new_seg.idx = new_seg, len(segs)
                    if k >= 0 and cur_team >= 0 and k != cur_team:
                        cur_team = -1
                cur_enc = e
                if cur.kind == "normal" and e is not None and not cur.frames:
                    cur.kind = "enc"                         # track bắt đầu ngay trong 1 lần gặp
                    cur.partners = set(enc_partners[e])
                if k >= 0:
                    cur_team = k
                cur.frames.append(f)
                where[(f, t)] = cur
                p = positions.get((f, t))
                if p is not None:
                    cur.pts.append((f, p[0], p[1]))
            cur.team = cur_team
            segs.append(cur)
        for f, t, label, prob in reads:
            s = where.get((f, t))
            if s is not None:                                # kể cả trong lúc gặp nhau: bỏ đi thì mất cả những lần
                s.reads.append((label, prob))                # đọc rất chắc (clip_3400: "3" prob 1.0 x 8 lần)
        for s in segs:
            v = TrackletVoter(**self.voter_kw)
            for label, prob in s.reads:
                v.add(0, label, prob)
            s.number = v.best(0)[0]
        return segs, where

    # ------------------------------------------------------------------ 2. ứng viên nối
    def _candidates(self, segs, by_frame):
        """by_frame: frame -> [(track, đội, x, y)] mọi cầu thủ có vị trí ở frame đó (để kiểm tra đám đông)."""
        self._geo = {}
        self._reid_pairs = set()
        self.n_crowd = self.n_reid = 0
        norm = [s for s in segs if s.pts and s.kind == "normal"]      # đoạn E không nối với ai
        ends = [(s, *_motion(s.pts, self.fps, self.k, True, self.vmax)) for s in norm]
        starts = sorted(((s, *_motion(s.pts, self.fps, self.k, False, self.vmax)) for s in norm), key=lambda r: r[1])
        start_f = np.array([r[1] for r in starts])
        cands = []
        for a, fa, pa, va in ends:
            lo = np.searchsorted(start_f, fa + 1)
            hi = np.searchsorted(start_f, fa + self.max_gap * self.fps, side="right")
            for b, fb, pb, vb in starts[lo:hi]:
                if b is a or b.frames[0] <= a.frames[-1] or b.role != a.role:
                    continue
                if a.team >= 0 and b.team >= 0 and a.team != b.team:
                    continue
                if a.number > 0 and b.number > 0 and a.number != b.number:
                    continue
                gap = (fb - fa) / self.fps
                pred = pa + va * min(gap, self.predict_s)
                r = min(self.r0 + self.spread * gap, self.r_max)
                d = float(np.linalg.norm(pred - pb))
                if d > r:
                    continue
                same_num = a.number > 0 and a.number == b.number
                # qua 1 lần gặp: A kết thúc khi bắt đầu gặp, B bắt đầu khi hết gặp, có chung đồng đội trong lần gặp
                via_enc = a.enc_end and b.enc_start and bool((a.partners | {a.tid}) & (b.partners | {b.tid}))
                # đám đông: 1 cầu thủ khác cùng đội (đang được track) cũng ở gần vị trí dự đoán -> không phân biệt được.
                # Qua lần gặp thì chính các đồng đội vừa gặp đương nhiên ở gần -> không tính họ (luật mập mờ lo việc
                # phân biệt A->B1 với A->B2).
                team = b.team if b.team >= 0 else a.team
                skip = (a.partners | b.partners | {a.tid}) if via_enc else set()
                near = [np.hypot(x - pred[0], y - pred[1]) for t, k, x, y in by_frame.get(fb, ())
                        if t != b.tid and t not in skip and (team < 0 or k in (team, -1))]
                via_reid = False
                if not same_num and near and min(near) <= d + self.crowd_margin:
                    rivals = [t for t, k, x, y in by_frame.get(fb, ())
                              if t != b.tid and t not in skip and (team < 0 or k in (team, -1))
                              and np.hypot(x - pred[0], y - pred[1]) <= d + self.crowd_margin]
                    if not self._reid_resolves(a, b, fb, rivals):
                        self.n_crowd += 1
                        continue
                    via_reid = True                              # chỉ ngoại hình phân định được -> 'uncertain'
                    self.n_reid += 1
                cost = d / r
                sa, sb = np.linalg.norm(va), np.linalg.norm(vb)
                if sa > self.moving and sb > self.moving:
                    cost += self.w_dir * (1 - float(va @ vb) / (sa * sb)) / 2
                if sa > 0 and sb > 0:                            # đổi vận tốc đột ngột (đứng yên -> nước rút)
                    cost += self.w_vel * min(float(np.linalg.norm(vb - va)) / self.vmax, 1.0)
                if same_num:
                    cost -= 0.5
                if a.team < 0 or b.team < 0:
                    cost += 0.2                                  # chưa rõ đội -> kém tin hơn
                cands.append((cost, a.idx, b.idx, same_num))
                if via_reid:
                    self._reid_pairs.add((a.idx, b.idx))
                self._geo[(a.idx, b.idx)] = (round(gap, 2), round(d, 1), round(r, 1), round(float(sa), 1),
                                             round(float(sb), 1), via_enc)
        return cands

    # ------------------------------------------------------------------ 2b'. ngoại hình (re-ID)
    def _app(self, tid, frames):
        """Embedding trung bình (L2) của track trên các frame cho trước, None nếu < 2 frame có embedding."""
        v = [self.emb[(f, tid)] for f in frames if (f, tid) in self.emb]
        if len(v) < 2:
            return None
        m = np.mean(np.asarray(v, np.float32), 0)
        return m / (np.linalg.norm(m) + 1e-9)

    def _reid_resolves(self, a, b, fb, rivals):
        """Đám đông: B và các đồng đội `rivals` đều gần vị trí dự đoán. Ngoại hình cuối A giống B rõ rệt hơn
        mọi rival (chênh > REID_MARGIN khoảng cách cosine) -> cho nối. Không có embedding -> False."""
        if self.emb is None:
            return False
        k = self.reid_frames
        ea, eb = self._app(a.tid, a.frames[-k:]), self._app(b.tid, b.frames[:k])
        if ea is None or eb is None:
            return False
        d_ab = 1 - float(ea @ eb)
        for t in rivals:
            fr = self._emb_frames.get(t)
            if fr is None:
                return False
            i = int(np.searchsorted(fr, fb))
            ec = self._app(t, fr[max(0, i - k // 2): i + k // 2 + 1])
            if ec is None or 1 - float(ea @ ec) < d_ab + self.reid_margin:
                return False
        return True

    # ------------------------------------------------------------------ 2b''. ghép toàn cục
    def _solve_global(self, pairs):
        """Ghép cặp (cuối đoạn -> đầu đoạn) tối ưu toàn cục: max Σ (LINK_ACCEPT - chi phí) bằng Hungarian trên
        từng thành phần liên thông của đồ thị ứng viên. Cặp mập mờ = bỏ nó đi thì phương án tốt nhất còn lại kém
        hơn < LINK_AMBIGUITY (cùng nghĩa với 'ứng viên thứ hai gần bằng' của bản tham lam, nhưng có tính tới việc
        ứng viên thứ hai đã thuộc về đoạn khác). -> ([(chi phí, a, b, cùng số)] được chọn, số cặp mập mờ)."""
        from scipy.optimize import linear_sum_assignment

        par = {}

        def find(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for _, a, b, _ in pairs:
            par[find(("o", a))] = find(("i", b))
        comps = defaultdict(list)
        for p in pairs:
            comps[find(("o", p[1]))].append(p)

        def best(W):
            r, c = linear_sum_assignment(W, maximize=True)
            keep = W[r, c] > 0
            return float(W[r, c][keep].sum()), list(zip(r[keep], c[keep]))

        chosen, n_ambig = [], 0
        for ps in comps.values():
            rows = sorted({p[1] for p in ps})
            cols = sorted({p[2] for p in ps})
            ri, ci = {a: i for i, a in enumerate(rows)}, {b: j for j, b in enumerate(cols)}
            W = np.zeros((len(rows), len(cols)))
            info = {}
            for c, a, b, sn in ps:
                W[ri[a], ci[b]] = self.accept - c
                info[(ri[a], ci[b])] = (c, a, b, sn)
            total, sel = best(W)
            for i, j in sel:
                c, a, b, sn = info[(i, j)]
                if not sn and len(ps) > 1:
                    W2 = W.copy()
                    W2[i, j] = 0
                    if total - best(W2)[0] < self.ambig:
                        n_ambig += 1
                        continue
                chosen.append((c, a, b, sn))
        return sorted(chosen), n_ambig

    # ------------------------------------------------------------------ 2a. phân định qua lần gặp
    def _encounter_groups(self, segs, cands, by_tid):
        """Gom các đoạn E chồng thời gian và có nhau trong danh sách đồng đội thành 1 lần gặp. Mỗi lần gặp:
        (A: track -> đoạn ngay trước, B: track -> đoạn ngay sau, track nào chuyển động gợi ý đã hoán ID).

        Chuyển động KHÔNG đủ tin để bác tracker: kiểm tra trên clip_3400, 2/2 lần "đổi ID theo chuyển động" đều sai
        (2 cầu thủ chồng nhau trên ảnh chỉ cách ~0.5 m trên sân, trong nhiễu phép chiếu). Nên tracker giữ được ID
        thì luôn giữ; chuyển động chỉ dùng để (1) đánh dấu rủi ro khi nó ủng hộ phương án hoán ID rõ rệt
        (ENC_SWAP_MARGIN), (2) nối lại các track bị tracker làm đứt trong lần gặp (bước 2c)."""
        from scipy.optimize import linear_sum_assignment

        self._ccost = {(a, b): (c, sn) for c, a, b, sn in cands if self._geo[(a, b)][-1]}
        encs = [s for s in segs if s.kind == "enc"]
        gp = {s.idx: s.idx for s in encs}

        def groot(i):
            while gp[i] != i:
                gp[i] = gp[gp[i]]
                i = gp[i]
            return i

        for i, s in enumerate(encs):
            for o in encs[i + 1:]:
                overlap = s.frames[0] <= o.frames[-1] + self.enc_gap and o.frames[0] <= s.frames[-1] + self.enc_gap
                if overlap and (o.tid in s.partners or s.tid in o.partners or o.tid == s.tid):
                    gp[groot(o.idx)] = groot(s.idx)
        members = defaultdict(set)
        for s in encs:
            members[groot(s.idx)].add(s.idx)

        groups = []
        for es in members.values():
            As, Bs = {}, {}                                    # track -> đoạn A / B
            for t in {segs[q].tid for q in es}:
                ss = by_tid[t]
                mine = [i for i, s in enumerate(ss) if s.idx in es]
                i0, i1 = min(mine), max(mine)
                if i0 > 0 and ss[i0 - 1].kind == "normal" and ss[i0 - 1].enc_end:
                    As[t] = ss[i0 - 1].idx
                if i1 + 1 < len(ss) and ss[i1 + 1].kind == "normal" and ss[i1 + 1].enc_start:
                    Bs[t] = ss[i1 + 1].idx
            suspect = set()
            keep = [t for t in As if t in Bs]
            if keep:
                al, bl = list(As.values()), list(Bs.values())
                cost = lambda a, b: self._ccost.get((a, b), (self.enc_unmatched * 2, False))[0]   # noqa: E731
                C = np.array([[cost(a, b) for b in bl] for a in al])
                keep_cost = sum(cost(As[t], Bs[t]) for t in keep)
                for t in keep:                                 # phương án tốt nhất KHÔNG giữ A_t -> B_t
                    M = C.copy()
                    M[al.index(As[t]), bl.index(Bs[t])] = 1e6
                    r, c = linear_sum_assignment(M)
                    rows = dict(zip(r, c))
                    alt = sum(min(M[al.index(As[k]), rows[al.index(As[k])]], self.enc_unmatched)
                              if al.index(As[k]) in rows else self.enc_unmatched for k in keep)
                    if alt < keep_cost - self.enc_margin:
                        suspect.add(t)
            groups.append((As, Bs, suspect))
        return groups

    def _relink(self, al, bl):
        """Hungarian trên các đoạn A / B còn trống của 1 lần gặp -> [(a, b, chi phí, cùng số áo)]."""
        from scipy.optimize import linear_sum_assignment

        if not al or not bl:
            return []
        big = 1e6
        C = np.array([[self._ccost.get((a, b), (big, False))[0] for b in bl] for a in al])
        r, c = linear_sum_assignment(C)
        return [(al[i], bl[j], float(C[i, j]), self._ccost[(al[i], bl[j])][1]) for i, j in zip(r, c) if C[i, j] < big]

    # ------------------------------------------------------------------ 3-4. ghép + định danh
    def run(self, dets, positions, team_of, roles, reads, emb=None):
        """dets: [(frame, track)], positions: {(frame, track): (x, y)} (m), team_of: {(frame, track): đội},
        roles: {track: role}, reads: [(frame, track, label, prob)],
        emb: {(frame, track): embedding re-ID} (tuỳ chọn) -> phân định cặp nối trong đám đông."""
        self.emb = emb or None
        self._emb_frames = {}
        if self.emb:
            fr = defaultdict(list)
            for f, t in self.emb:
                fr[t].append(f)
            self._emb_frames = {t: np.array(sorted(v)) for t, v in fr.items()}
        by_frame = defaultdict(list)
        for (f, t), (x, y) in positions.items():
            if roles.get(t, "player") != "referee":
                by_frame[f].append((t, team_of.get((f, t), -1), x, y))
        segs, where = self._segments(dets, positions, team_of, roles, reads, by_frame)
        cands = sorted(self._candidates(segs, by_frame))
        best_out, best_in = defaultdict(list), defaultdict(list)
        for c, a, b, _ in cands:
            best_out[a].append(c)
            best_in[b].append(c)

        parent = list(range(len(segs)))
        votes = [defaultdict(float) for _ in segs]               # phiếu số áo theo chuỗi (gốc union-find)
        tmp = TrackletVoter(**self.voter_kw)
        for s in segs:
            for label, prob in s.reads:
                tmp.add(s.idx, label, prob)
            for n, w in tmp.weights(s.idx).items():
                votes[s.idx][n] += w

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def chain_number(r):
            if not votes[r]:
                return -1
            n, w = max(votes[r].items(), key=lambda kv: kv[1])
            return n if w > self.voter_kw["sum_threshold"] else -1

        succ, pred = {}, {}
        risky = {}                                               # (a, b) -> liên kết qua lần gặp mà không phân định rõ
        self.links = []
        n_ambig = n_conflict = n_overlap = 0
        # cannot-link: 1 cầu thủ không thể ở 2 nơi trong cùng 1 frame -> 2 chuỗi có frame chung không bao giờ được nối
        occupied = {s.idx: set(s.frames) for s in segs}          # gốc chuỗi -> các frame chuỗi có mặt

        def union(ra, rb):
            parent[rb] = ra
            occupied[ra] |= occupied.pop(rb)
            for n, w in votes[rb].items():
                votes[ra][n] += w

        def link(a, b, c, is_risky=False, decision=None):
            nonlocal n_conflict, n_overlap
            ra, rb = root(a), root(b)
            if ra == rb or a in succ or b in pred:
                return False
            na, nb = chain_number(ra), chain_number(rb)
            if na > 0 and nb > 0 and na != nb:
                n_conflict += 1
                return False
            if not occupied[ra].isdisjoint(occupied[rb]):
                n_overlap += 1
                return False
            succ[a], pred[b] = b, a
            risky[(a, b)] = is_risky
            self.links.append({"from_track": segs[a].tid, "from_frame": segs[a].frames[-1], "to_track": segs[b].tid,
                               "to_frame": segs[b].frames[0], "cost": round(c, 3),
                               **dict(zip(("gap_s", "dist_m", "radius_m", "speed_from", "speed_to", "via_encounter"),
                                          self._geo[(a, b)])),
                               **({"decision": decision} if decision else {})})
            union(ra, rb)
            return True

        by_tid = defaultdict(list)
        for s in segs:
            by_tid[s.tid].append(s)
        for ss in by_tid.values():
            ss.sort(key=lambda s: s.frames[0])

        # đoạn E (đang gặp đồng đội): tin tracker -> thuộc về chuỗi của đoạn liền kề CÙNG track.
        #   (1) trước khi nối: track kết thúc trong lần gặp -> theo đoạn trước; bắt đầu trong lần gặp -> theo đoạn sau
        #       (làm sớm để cannot-link thấy cả các frame lúc gặp: 2 track cùng có mặt trong lần gặp = 2 người);
        #   (2) sau khi nối: có cả đoạn trước và sau, cùng 1 chuỗi -> gắn; 2 chuỗi khác nhau -> không biết, bỏ trống
        n_enc_attached, e_prev = 0, {}

        def attach_enc(both):
            nonlocal n_enc_attached
            for ss in by_tid.values():
                for i, s in enumerate(ss):
                    if s.kind != "enc" or s.idx in e_prev:
                        continue
                    pa = ss[i - 1] if i > 0 and ss[i - 1].kind == "normal" else None
                    nb = ss[i + 1] if i + 1 < len(ss) and ss[i + 1].kind == "normal" else None
                    if both:
                        anchor = pa if pa is not None and nb is not None and root(pa.idx) == root(nb.idx) else None
                    else:
                        anchor = (pa or nb) if (pa is None) != (nb is None) else None
                    if anchor is None:
                        continue
                    ra, rs = root(anchor.idx), root(s.idx)
                    if ra != rs and occupied[ra].isdisjoint(occupied[rs]):
                        union(ra, rs)
                        e_prev[s.idx] = anchor.idx
                        n_enc_attached += 1

        if self.enc_split:
            attach_enc(both=False)

        # 2a. qua lần gặp, tracker giữ được ID: luôn giữ
        self.enc_decisions = {"keep": 0, "keep_suspect": 0, "relink": 0, "relink_same_number": 0}
        groups = self._encounter_groups(segs, cands, by_tid) if self.enc_split else []
        for As, Bs, suspect in groups:
            for t in As:
                if t in Bs:
                    c = self._ccost.get((As[t], Bs[t]), (np.nan, False))[0]
                    self._geo.setdefault((As[t], Bs[t]), (None,) * 5 + (True,))
                    d = "keep_suspect" if t in suspect else "keep"
                    if link(As[t], Bs[t], c, t in suspect and self.enc_motion_suspect, d):
                        self.enc_decisions[d] += 1
        # 2c. track bị tracker làm đứt trong lần gặp: nối lại phần còn trống bằng chuyển động; chỉ chắc khi số áo
        #     xác nhận (chuyển động lúc chồng nhau không đáng tin). Cannot-link chặn nối 2 track cùng có mặt.
        for As, Bs, _ in groups:
            al = [a for a in As.values() if a not in succ]
            bl = [b for b in Bs.values() if b not in pred]
            for a, b, c, same in self._relink(al, bl):
                d = "relink_same_number" if same else "relink"
                if link(a, b, c, not same, d):
                    self.enc_decisions[d] += 1

        # 2b. các cặp còn lại: ghép toàn cục (mặc định) hoặc tham lam theo chi phí, bỏ cặp mập mờ
        if self.solver == "global":
            free = [p for p in cands if p[1] not in succ and p[2] not in pred and not self._geo[(p[1], p[2])][-1]]
            chosen, n_ambig = self._solve_global(free)
            for c, a, b, _ in chosen:
                link(a, b, c, (a, b) in self._reid_pairs, "reid" if (a, b) in self._reid_pairs else None)
        else:
            for c, a, b, same_num in cands:
                if a in succ or b in pred or self._geo[(a, b)][-1]:
                    continue
                if not same_num and min(_runner_up(best_out[a], c), _runner_up(best_in[b], c)) - c < self.ambig:
                    n_ambig += 1                                 # A (hoặc B) có ứng viên khác gần như ngang
                    continue
                link(a, b, c, (a, b) in self._reid_pairs, "reid" if (a, b) in self._reid_pairs else None)

        if self.enc_split:
            attach_enc(both=True)

        # định danh: chuỗi có số -> (đội, số); chuỗi không số -> id riêng
        chain_team = defaultdict(lambda: np.zeros(2))
        for s in segs:
            if s.team >= 0:
                chain_team[root(s.idx)][s.team] += len(s.frames)
        def team_of_chain(r):
            tv = chain_team.get(r)
            return int(tv.argmax()) if tv is not None and tv.sum() > 0 else -1

        # 1 cầu thủ không thể ở 2 nơi trong cùng frame: các chuỗi cùng (đội, số) mà chồng nhau về thời gian
        # -> chuỗi nhiều phiếu nhất giữ số, chuỗi còn lại mất số (thường do PARSeq đọc nhầm, vd bịa "27")
        chain_frames = defaultdict(set)
        for s in segs:
            chain_frames[root(s.idx)].update(s.frames)
        final_num = {r: chain_number(r) for r in chain_frames}
        groups = defaultdict(list)
        for r, n in final_num.items():
            if n > 0:
                groups[(team_of_chain(r), n)].append(r)
        n_demoted = 0
        for (_, n), rs in groups.items():
            kept = []
            for r in sorted(rs, key=lambda r: -votes[r][n]):
                if any(chain_frames[r] & chain_frames[k] for k in kept):
                    final_num[r] = -1
                    n_demoted += 1
                else:
                    kept.append(r)

        # nguồn số áo của từng đoạn: đi dọc chuỗi (A -> B -> C ...) tới đoạn gần nhất có đọc được số đó;
        # đường đi qua 1 liên kết qua lần gặp không phân định rõ -> 'uncertain'
        seg_w = {s.idx: tmp.weights(s.idx) for s in segs}
        # "có đọc": đoạn tự có >= ~1 lần đọc chắc chắn (0.5 ~ 1 lần prob 0.85 x bias 0.61) — không đòi đủ ngưỡng
        # chốt số của cả track, vì cắt ở lần gặp làm mỗi đoạn ngắn hơn
        saw = lambda q, num: seg_w[q].get(num, 0) >= 0.5      # noqa: E731
        hops = {}                                             # đoạn -> số liên kết rủi ro tới lần đọc gần nhất
        for s in segs:
            if s.idx in pred or s.kind != "normal":
                continue
            path = [s.idx]                                    # s là đầu chuỗi
            while path[-1] in succ:
                path.append(succ[path[-1]])
            num = final_num[root(s.idx)]
            if num <= 0:
                continue
            best = [np.inf] * len(path)
            for direction in (range(len(path)), range(len(path) - 1, -1, -1)):
                h = np.inf
                prev = None
                for i in direction:
                    if prev is not None and h < np.inf:
                        a, b = (path[prev], path[i]) if prev < i else (path[i], path[prev])
                        h += risky.get((a, b), False)
                    if saw(path[i], num):
                        h = 0
                    best[i] = min(best[i], h)
                    prev = i
            for i, q in enumerate(path):
                hops[q] = best[i]

        out, players = {}, {}
        for s in segs:
            r = root(s.idx)
            num = final_num[r]
            team = team_of_chain(r)
            if num > 0:
                tag = "AB"[team] if team >= 0 else ("GK" if s.role == "goalkeeper" else "?")
                pid = f"{tag}{num}"
            else:
                pid = f"c{r}"
            if num <= 0:
                src = ""
            elif s.kind == "enc" and not saw(s.idx, num):     # đoạn E gắn lại: tin như đoạn liền kề nó
                q = e_prev.get(s.idx)
                src = "linked" if q is not None and hops.get(q, np.inf) == 0 else "uncertain"
            elif saw(s.idx, num):
                src = "read"                                  # chính đoạn này có lần đọc ra số đó
            else:
                src = "linked" if hops.get(s.idx, np.inf) == 0 else "uncertain"
            for f in s.frames:
                out[(f, s.tid)] = (pid, num, src)
            p = players.setdefault(pid, {"team": team, "number": num, "role": s.role, "tracks": set(),
                                         "segments": 0, "n_frames": 0, "first_frame": s.frames[0],
                                         "last_frame": s.frames[-1], "n_reads": 0})
            p["tracks"].add(s.tid)
            p["segments"] += 1
            p["n_frames"] += len(s.frames)
            p["n_reads"] += len(s.reads)
            p["first_frame"] = min(p["first_frame"], s.frames[0])
            p["last_frame"] = max(p["last_frame"], s.frames[-1])

        # kiểm tra: cùng 1 player_id xuất hiện 2 lần trong cùng frame = nối sai hoặc đọc nhầm số
        per_frame = defaultdict(lambda: defaultdict(int))
        for (f, t), (pid, num, _) in out.items():
            if num > 0:
                per_frame[f][pid] += 1
        dup = sum(1 for d in per_frame.values() for c in d.values() if c > 1)

        numbered = [pid for pid, p in players.items() if p["number"] > 0]
        stats = {"segments": len(segs), "links": len(succ), "skipped_ambiguous": n_ambig,
                 "skipped_crowd": self.n_crowd, "crowd_resolved_by_reid": self.n_reid,
                 "links_reid": sum(1 for (a, b) in succ.items() if (a, b) in self._reid_pairs),
                 "solver": self.solver, "encounters": self.n_enc,
                 "encounter_segments": sum(1 for s in segs if s.kind == "enc"),
                 "encounter_links": sum(self.enc_decisions.values()), "encounter_decisions": self.enc_decisions,
                 "encounter_segments_attached": n_enc_attached,
                 "dets_uncertain": sum(1 for v in out.values() if v[2] == "uncertain"),
                 "skipped_number_conflict": n_conflict, "skipped_same_frame": n_overlap,
                 "demoted_same_frame": n_demoted,
                 "players_with_number": len(numbered),
                 "dets_read": sum(1 for v in out.values() if v[2] == "read"),
                 "dets_linked": sum(1 for v in out.values() if v[2] == "linked"),
                 "dets_total": len(out), "dup_frame_ids": dup}
        for p in players.values():
            p["tracks"] = sorted(p["tracks"])
        return out, players, stats
