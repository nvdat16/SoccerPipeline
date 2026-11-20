"""Pipeline: detect + track cầu thủ (YOLO + ByteTrack) -> nhận dạng số áo theo tracklet.

    python run.py --source video.mp4 --output outputs/

Các bước, mỗi frame:
  1. YOLO detect người và ByteTrack gán track id.
  2. Mỗi FRAME_STRIDE frame, với từng cầu thủ đủ lớn:
     a. Legibility classifier (ResNet34) trên crop cầu thủ: có thấy số áo không?
     b. Lấy keypoints vai/hông (từ YOLO-pose) rồi cắt vùng thân áo.
     c. PARSeq đọc số trên crop thân áo.
  3. Hết video: gộp các lần đọc theo track (bỏ phiếu có trọng số) để ra 1 số áo / track.
  4. Chia đội (--teams, mặc định bật): histogram màu thân áo lấy mẫu trong lúc chạy -> KMeans 2 cụm.
  5. Render video kết quả và xuất JSON/CSV.

Với --minimap, thêm mỗi frame: phát hiện cắt cảnh (reset tracker + phép chiếu), pitch keypoints ->
homography ảnh->sân, chiếu chân cầu thủ lên sân (mét); cuối cùng làm mượt vị trí và vẽ minimap.
"""
import argparse
import csv
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import config
from pipeline.aggregator import TrackletVoter
from pipeline.device import pick_device
from pipeline.identity import IdentityLinker
from pipeline.legibility import LegibilityModel
from pipeline.minimap import MinimapRenderer, draw_debug, overlay, smooth_positions
from pipeline.pitch import PitchCalibrator, PitchConfig, project
from pipeline.profiler import StepTimer
from pipeline.shots import ShotDetector
from pipeline.str_recognizer import JerseyNumberRecognizer
from pipeline.tiling import OwnTracker, tiled_detect
from pipeline.team import TeamClassifier, text_ink
from pipeline.torso import match_pose_to_box, torso_box, torso_from_bbox

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video, ảnh, hoặc folder chứa video/ảnh")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--models-dir", default=str(config.MODELS_DIR))
    ap.add_argument("--det-model", default=config.DET_MODEL)
    ap.add_argument("--pose-model", default=config.POSE_MODEL)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="class của cầu thủ, theo id hoặc tên (vd: '1 2' hoặc 'player goalkeeper'). "
                         f"Mặc định: tự chọn các class có tên trong {config.PLAYER_CLASS_NAMES}")
    ap.add_argument("--tracker", default=config.TRACKER,
                    help="file yaml của tracker (mặc định trackers/bytetrack_soccer.yaml); "
                         "trackers/botsort_soccer.yaml (thêm GMC) | bytetrack.yaml | botsort.yaml")
    ap.add_argument("--conf", type=float, default=config.DET_CONF)
    ap.add_argument("--imgsz", type=int, default=config.DET_IMGSZ)
    ap.add_argument("--tile", action=argparse.BooleanOptionalAction, default=config.TILE,
                    help=f"detect thêm trên lưới tile {config.TILE_GRID} ở vùng xa camera {config.TILE_REGION} "
                         "để bắt cầu thủ nhỏ (chậm hơn; đo trước bằng tools/far_report.py --video)")
    ap.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    ap.add_argument("--min-box-height", type=int, default=config.MIN_BOX_HEIGHT)
    ap.add_argument("--legibility-th", type=float, default=config.LEGIBILITY_THRESHOLD)
    ap.add_argument("--no-legibility", action="store_true",
                    help="bỏ qua legibility classifier, đọc số trên mọi cầu thủ (chậm hơn, nhiều nhiễu hơn)")
    ap.add_argument("--torso-mode", choices=["pose", "bbox"], default=config.TORSO_MODE,
                    help="pose: cắt thân áo theo keypoint vai/hông; bbox: cắt theo tỉ lệ bbox, bỏ hẳn model pose")
    ap.add_argument("--torso-ratio", type=float, nargs=4, default=list(config.TORSO_BBOX_RATIO),
                    metavar=("X1", "Y1", "X2", "Y2"), help="tỉ lệ thân áo trong bbox (cho --torso-mode bbox)")
    ap.add_argument("--kpt-conf", type=float, default=config.KPT_CONF)
    ap.add_argument("--max-reads", type=int, default=config.MAX_READS_PER_TRACK)
    ap.add_argument("--sum-th", type=float, default=config.SUM_THRESHOLD)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = toàn bộ video")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--no-video", action="store_true", help="không render video kết quả")
    ap.add_argument("--save-crops", action="store_true", help="lưu crop thân áo + số đọc được (debug)")
    ap.add_argument("--link", action=argparse.BooleanOptionalAction, default=True,
                    help="(cần --minimap) nối các đoạn track thành 1 cầu thủ theo vị trí/vận tốc/hướng trên sân, "
                         "truyền số áo đọc rõ sang các đoạn chưa đọc được")
    ap.add_argument("--enc-split", action=argparse.BooleanOptionalAction, default=config.ENC_SPLIT,
                    help="(--link) cắt track khi 2 đồng đội gặp nhau (< ENC_R m), quyết định lại ai là ai bằng "
                         "chuyển động; số áo chỉ có nhờ nối qua lần gặp được đánh dấu 'uncertain'")
    ap.add_argument("--link-solver", choices=["global", "greedy"], default=config.LINK_SOLVER,
                    help="(--link) global: ghép Hungarian tối ưu toàn cục; greedy: tham lam theo chi phí (bản cũ)")
    ap.add_argument("--reid", action=argparse.BooleanOptionalAction, default=config.REID,
                    help="(--link) embedding ngoại hình OSNet để phân định cặp nối trong đám đông "
                         f"(cần <models-dir>/{config.REID_FILE}); số áo truyền qua đó = 'uncertain'")
    ap.add_argument("--teams", action=argparse.BooleanOptionalAction, default=True,
                    help="chia 2 đội theo màu áo (histogram HSV + KMeans), tô màu bbox/minimap theo đội")

    mm = ap.add_argument_group("minimap")
    mm.add_argument("--minimap", action=argparse.BooleanOptionalAction, default=False,
                    help="dựng minimap 2D: vị trí (mét) của cầu thủ trên sân, vẽ ở góc video + positions.csv")
    mm.add_argument("--calib", choices=["pnlcalib", "roboflow"], default=config.CALIB_BACKEND,
                    help="pnlcalib: HRNet điểm + vạch có tên (chính xác, chậm hơn); "
                         "roboflow: YOLOv8-pose 32 điểm (nhanh, hay nhầm vòng cấm thành giữa sân)")
    mm.add_argument("--pnl-refine", action=argparse.BooleanOptionalAction, default=config.PNL_REFINE,
                    help="PnLCalib: tinh chỉnh homography theo các vạch sân")
    mm.add_argument("--pitch-model", default=None, help=f"(--calib roboflow) mặc định <models-dir>/{config.PITCH_FILE}")
    mm.add_argument("--pitch-size", type=float, nargs=2, default=list(config.PITCH_SIZE),
                    metavar=("LENGTH", "WIDTH"), help="kích thước sân (m)")
    mm.add_argument("--pitch-kpt-conf", type=float, default=config.PITCH_KPT_CONF)
    mm.add_argument("--pitch-every", type=int, default=None,
                    help=f"chạy calibration mỗi N frame, giữa các lần nối bằng optical flow "
                         f"(mặc định {config.PITCH_EVERY})")
    mm.add_argument("--pitch-debug", action="store_true",
                    help="vẽ vạch sân chiếu ngược + keypoints lên video (kiểm tra homography)")
    mm.add_argument("--reset-on-cut", action=argparse.BooleanOptionalAction, default=None,
                    help="reset tracker ở mỗi cắt cảnh (mặc định: bật khi --minimap)")
    mm.add_argument("--minimap-video", action="store_true", help="xuất thêm video chỉ có minimap")
    mm.add_argument("--minimap-pos", default=config.MINIMAP_POS)
    a = ap.parse_args()
    if a.reset_on_cut is None:
        a.reset_on_cut = a.minimap
    if a.pitch_every is None:
        a.pitch_every = config.PITCH_EVERY[a.calib]
    return a


def role_of(name: str) -> str:
    n = str(name).lower()
    if n in ("goalkeeper", "goalie"):
        return "goalkeeper"
    if n in config.REFEREE_CLASS_NAMES:
        return "referee"
    return "player"


def reset_tracker(detector):
    """Xoá mọi track đang có (sau cắt cảnh, vị trí bbox cũ không còn liên quan).
    Một số bản ultralytics đặt lại ID về 1 khi reset -> run.py tự cộng offset để ID không trùng."""
    for t in getattr(getattr(detector, "predictor", None), "trackers", None) or []:
        t.reset()


class PlayerJerseyPipeline:
    def __init__(self, args):
        self.args = args
        self.device = pick_device(args.device)
        models_dir = Path(args.models_dir)
        print(f"[init] device={self.device}")
        self.legibility = LegibilityModel(models_dir / config.LEGIBILITY_FILE, self.device, args.legibility_th)
        self.recognizer = JerseyNumberRecognizer(models_dir / config.STR_FILE, self.device)
        det = YOLO(args.det_model)
        self.det_is_pose = det.task == "pose"
        self.class_names = det.names
        args.classes = resolve_classes(det.names, args.classes)
        self.read_classes = set(args.classes)          # chỉ đọc số áo các class này
        if args.minimap:                               # minimap vẽ cả trọng tài (không đọc số)
            args.classes = sorted(set(args.classes) | {i for i, n in det.names.items()
                                                       if str(n).lower() in config.REFEREE_CLASS_NAMES})
        print(f"[init] detector={args.det_model} (task={det.task}) -> track classes "
              f"{ {i: det.names[i] for i in args.classes} }")
        self.pitch_cfg = PitchConfig(length=args.pitch_size[0], width=args.pitch_size[1])
        self.calib = None
        if args.minimap:
            pm, pnl = None, None
            if args.calib == "pnlcalib":
                from pipeline.calib_pnl import PnLCalibModel
                if tuple(args.pitch_size) != (105.0, 68.0):
                    print("[warn] PnLCalib dùng sân 105x68 cố định (theo SoccerNet) -> bỏ qua --pitch-size")
                    self.pitch_cfg = PitchConfig()
                kp_w, ln_w = models_dir / config.PNL_KP_FILE, models_dir / config.PNL_LINE_FILE
                missing = [str(p) for p in (kp_w, ln_w) if not p.exists()]
                if missing:
                    raise SystemExit(f"Không có weights PnLCalib {missing}. Chạy `python download_models.py`.")
                pnl = PnLCalibModel(kp_w, ln_w, self.device, refine_lines=args.pnl_refine)
                desc = f"PnLCalib ({kp_w.name}, {ln_w.name}, refine={args.pnl_refine})"
            else:
                pm = Path(args.pitch_model) if args.pitch_model else models_dir / config.PITCH_FILE
                if not pm.exists():
                    raise SystemExit(f"Không có model pitch {pm}. Chạy `python download_models.py` "
                                     f"hoặc truyền --pitch-model.")
                desc = f"Roboflow 32 điểm ({pm})"
            self.calib = PitchCalibrator(
                pm, self.device, self.pitch_cfg, imgsz=config.PITCH_IMGSZ, det_conf=config.PITCH_DET_CONF,
                kpt_conf=args.pitch_kpt_conf, ransac_m=config.PITCH_RANSAC_M,
                max_reproj_m=config.PITCH_MAX_REPROJ_M, max_jump_m=config.PITCH_MAX_JUMP_M,
                ema=config.PITCH_EMA, max_flow_frames=config.PITCH_MAX_FLOW_FRAMES,
                # switch_after tính theo số lần đo -> quy đổi để luôn ~PITCH_SWITCH_AFTER frame
                switch_after=max(3, -(-config.PITCH_SWITCH_AFTER // max(args.pitch_every, 1))),
                pnl=pnl, max_reproj_px=config.PNL_MAX_REPROJ_PX)
            print(f"[init] minimap: {desc}, sân {self.pitch_cfg.length:g}x{self.pitch_cfg.width:g} m, "
                  f"mỗi {args.pitch_every} frame, reset tracker ở cắt cảnh={args.reset_on_cut}")
        self.reid = None
        if args.reid and args.minimap and args.link:
            from pipeline.reid import OSNetEmbedder
            rw = models_dir / config.REID_FILE
            if not rw.exists():
                raise SystemExit(f"Không có weights re-ID {rw}. Chạy `python download_models.py`.")
            self.reid = OSNetEmbedder(rw, self.device)
            print(f"[init] re-ID: OSNet-AIN ({rw.name}), box >= {config.REID_MIN_BOX_HEIGHT}px, mỗi {args.stride} frame")
        if args.tile and self.det_is_pose and args.torso_mode == "pose":
            raise SystemExit("--tile chưa hỗ trợ lấy keypoint từ detector pose: dùng --torso-mode bbox")
        if args.tile:
            print(f"[init] tile: lưới {config.TILE_GRID}, vùng {config.TILE_REGION}, phóng to x{config.TILE_UPSCALE}, "
                  f"chỉ thêm box < {config.TILE_MAX_H}px")
        if args.torso_mode == "bbox":
            self.pose = None
            print(f"[init] torso-mode=bbox (không dùng pose), ratio={tuple(args.torso_ratio)}")
        elif self.det_is_pose:
            self.pose = None
        else:
            self.pose = YOLO(args.pose_model)
            print(f"[init] pose model={args.pose_model}")

    # ------------------------------------------------------------------ pass 1
    def analyse(self, frames, total, prof: StepTimer, crops_dir=None):
        a = self.args
        detector = YOLO(a.det_model)  # model mới cho mỗi video -> reset tracker
        yolo_kw = dict(conf=a.conf, imgsz=a.imgsz, classes=a.classes, device=self.device, verbose=False)
        voter = TrackletVoter(config.FILTER_THRESHOLD, a.sum_th, config.USE_BIAS)
        teamer = TeamClassifier(config.TEAM_FIT_SAMPLES, config.TEAM_MIN_SAMPLES, config.TEAM_OUTLIER_MAD,
                                config.TEAM_SMOOTH) if a.teams else None
        # [(frame_idx, track_id, x1, y1, x2, y2, det_conf, cls, shot_id, x_m, y_m, h_status)]
        frame_log = []
        read_log = []             # [(frame_idx, track_id, label, prob)] mọi lần PARSeq đọc
        span = {}                 # track_id -> [first_frame, last_frame]
        stats = defaultdict(lambda: {"attempts": 0, "legible": 0, "no_pose": 0})
        shots = ShotDetector() if (a.minimap or a.reset_on_cut) else None
        if self.calib is not None:
            self.calib.reset()
        geo = {"homs": {}, "shots": 0, "status": defaultdict(int), "err_m": [], "err_px": [],
               "projected": 0, "outside": 0, "cut_off": 0, "players": 0,
               "motion": {}, "shot_of": {}, "shape": None, "emb": {}}
        shot_id, shot_start, id_offset, max_id = 0, 0, 0, 0
        own = OwnTracker(a.tracker) if a.tile else None   # --tile: tự gọi tracker với detection đã gộp full + tile
        geo["tile_extra"] = 0

        for idx, frame in tqdm(timed_frames(frames, prof), total=total, desc="analyse"):
            is_cut = False
            if shots is not None:
                with prof("2a. shot detect", n_items=1):
                    is_cut = shots.update(frame)
                shot_id = shots.shot_id
                if is_cut:
                    shot_start = idx
                    if a.reset_on_cut and idx > 0:
                        if own is not None:
                            own.reset()
                        else:
                            reset_tracker(detector)
                        id_offset = max_id
            tile_ms = 0.0
            with prof("2. detect+track", n_items=1):
                t0 = time.perf_counter()
                if own is None:
                    r = detector.track(frame, persist=True, tracker=a.tracker, **yolo_kw)[0]
                else:
                    r = detector.predict(frame, **yolo_kw)[0]
                    full = ((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())
                            if r.boxes is not None else (np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
                    t1 = time.perf_counter()
                    tb, tc, tk, n_extra = tiled_detect(
                        detector, frame, None, a.conf, a.classes, self.device, grid=config.TILE_GRID,
                        overlap=config.TILE_OVERLAP, region=config.TILE_REGION, full=full, ios_th=config.TILE_IOS,
                        max_h=config.TILE_MAX_H, upscale=config.TILE_UPSCALE)
                    tile_ms = (time.perf_counter() - t1) * 1000
                    geo["tile_extra"] += n_extra
                    tracks_arr = own.update(tb, tc, tk, frame)
                total_ms = (time.perf_counter() - t0) * 1000
            sp = r.speed  # ms, ultralytics tự đo
            prof.add("   ├ yolo preprocess", sp.get("preprocess", 0))
            prof.add("   ├ yolo inference", sp.get("inference", 0))
            prof.add("   ├ yolo postprocess", sp.get("postprocess", 0))
            if own is not None:
                prof.add("   ├ tile detect", tile_ms)
            prof.add("   └ tracker", max(0.0, total_ms - tile_ms - sum(sp.get(k, 0) for k in
                                                                     ("preprocess", "inference", "postprocess"))))
            if own is not None:
                if len(tracks_arr):
                    boxes = tracks_arr[:, :4].astype(np.float32)
                    ids = tracks_arr[:, 4].astype(int) + id_offset
                    confs = tracks_arr[:, 5].astype(np.float32)
                    clss = tracks_arr[:, 6].astype(int)
                    max_id = max(max_id, int(ids.max()))
                else:
                    boxes, ids, confs, clss = np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros(0, int)
                kpts = None
            elif r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.int().cpu().numpy() + id_offset
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.int().cpu().numpy()
                kpts = r.keypoints.data.cpu().numpy() if self.det_is_pose and r.keypoints is not None else None
                max_id = max(max_id, int(ids.max()))
            else:
                boxes, ids, confs, clss, kpts = np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros(0, int), None

            # --- minimap: homography ảnh -> sân, chiếu chân cầu thủ (mọi frame, kể cả không có cầu thủ)
            pos, status = [None] * len(boxes), ""
            if self.calib is not None:
                with prof("2b. pitch calib", n_items=1):
                    run_model = (idx - shot_start) % max(a.pitch_every, 1) == 0
                    H, status, info = self.calib.update(frame, is_cut, boxes, run_model=run_model)
                if "model_ms" in info:
                    prof.add("   ├ pitch keypoints", info["model_ms"])
                pos = self._project_feet(H, boxes, frame.shape, geo)
                geo["status"][status] += 1
                geo["motion"][idx], geo["shot_of"][idx], geo["shape"] = info.get("motion"), shot_id, frame.shape
                if status == "kp" and "err_m" in info:
                    geo["err_m"].append(info["err_m"])
                if status == "kp" and "rep_err_px" in info:
                    geo["err_px"].append(info["rep_err_px"])
                dbg = None
                if a.pitch_debug:
                    dbg = {k: info[k] for k in ("kp_xy", "kp_conf", "n_kp", "n_lines", "n_inliers", "err_m",
                                                "rep_err_px", "rejected") if k in info}
                    dbg["shot"] = shot_id
                geo["homs"][idx] = (H, status, dbg)

            for b, t, c, cl, p in zip(boxes, ids, confs, clss, pos):
                frame_log.append((idx, int(t), *[int(v) for v in b], float(c), int(cl), shot_id,
                                  None if p is None else round(float(p[0]), 2),
                                  None if p is None else round(float(p[1]), 2), status))
                span.setdefault(int(t), [idx, idx])[1] = idx

            if not len(boxes) or idx % a.stride:
                continue

            # --- mẫu màu áo để chia đội (mọi cầu thủ đủ lớn, không phụ thuộc legibility)
            if teamer is not None:
                with prof("5b. team color sample", n_items=1):
                    for b, t, cl in zip(boxes, ids, clss):
                        if b[3] - b[1] >= config.TEAM_MIN_BOX_HEIGHT and int(cl) in self.read_classes:
                            tb = torso_from_bbox(b, frame.shape, config.TEAM_BBOX_RATIO)
                            if tb is not None:
                                teamer.add(int(t), idx, frame[tb[1]:tb[3], tb[0]:tb[2]])

            # --- embedding ngoại hình (--reid) cho bước nối định danh
            if self.reid is not None:
                sel = [i for i, (b, cl) in enumerate(zip(boxes, clss))
                       if b[3] - b[1] >= config.REID_MIN_BOX_HEIGHT and int(cl) in self.read_classes]
                if sel:
                    with prof("5c. re-ID embed", n_items=len(sel)):
                        h, w = frame.shape[:2]
                        crops = [frame[max(int(boxes[i][1]), 0):min(int(boxes[i][3]), h),
                                       max(int(boxes[i][0]), 0):min(int(boxes[i][2]), w)] for i in sel]
                        for i, e in zip(sel, self.reid(crops)):
                            geo["emb"][(idx, int(ids[i]))] = e

            # --- chọn cầu thủ cần đọc số ở frame này
            cand = [i for i, (b, t) in enumerate(zip(boxes, ids))
                    if b[3] - b[1] >= a.min_box_height and stats[int(t)]["attempts"] < a.max_reads
                    and int(clss[i]) in self.read_classes]
            if not cand:
                continue
            if a.no_legibility:
                leg_prob = np.ones(len(cand))
            else:
                with prof("3. legibility", n_items=len(cand)):
                    player_crops = [frame[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2])]
                                    for i in cand]
                    leg_prob = self.legibility.predict(player_crops)

            legible = [i for i, p in zip(cand, leg_prob) if p > a.legibility_th]
            for i in cand:
                stats[int(ids[i])]["attempts"] += 1
            if not legible:
                continue

            # --- keypoints -> crop thân áo
            bbox_mode = a.torso_mode == "bbox"
            if kpts is None and not bbox_mode:
                with prof("4. pose (model riêng)", n_items=1):
                    pr = self.pose(frame, imgsz=a.imgsz, device=self.device, verbose=False)[0]
                    p_boxes = pr.boxes.xyxy.cpu().numpy() if pr.boxes is not None else np.zeros((0, 4))
                    p_kpts = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None else None
            torso_crops, owners = [], []
            t_crop = time.perf_counter()
            for i in legible:
                tid = int(ids[i])
                stats[tid]["legible"] += 1
                if bbox_mode:
                    tb = torso_from_bbox(boxes[i], frame.shape, a.torso_ratio)
                else:
                    if kpts is not None:
                        k = kpts[i]
                    else:
                        j = match_pose_to_box(boxes[i], p_boxes)
                        k = p_kpts[j] if j >= 0 and p_kpts is not None else None
                    tb = torso_box(k, frame.shape, a.kpt_conf, config.TORSO_PADDING)
                if tb is None:
                    stats[tid]["no_pose"] += 1
                    continue
                x1, y1, x2, y2 = tb
                torso_crops.append(frame[y1:y2, x1:x2])
                owners.append(tid)
            prof.add("5. torso crop", (time.perf_counter() - t_crop) * 1000, n_items=len(legible))

            # --- PARSeq đọc số
            if not torso_crops:
                continue
            with prof("6. STR (PARSeq)", n_items=len(torso_crops)):
                preds = self.recognizer.predict(torso_crops)
            for tid, crop, (label, prob) in zip(owners, torso_crops, preds):
                voter.add(tid, label, prob)
                read_log.append((idx, tid, label, prob))
                if crops_dir is not None:
                    with prof("   save crops (debug)", n_items=1):
                        cv2.imwrite(str(crops_dir / f"t{tid:04d}_f{idx:06d}_{label or 'x'}_{prob:.2f}.jpg"), crop)

        geo["shots"] = shots.shot_id + 1 if shots is not None else 1
        geo["reads"] = read_log
        return voter, frame_log, span, stats, geo, teamer

    def _fill_gaps(self, frame_log, geo):
        """Frame chưa có H trong shot -> lan H qua chuyển động camera (PitchCalibrator.fill_gaps), chiếu lại chân
        cầu thủ ở các frame đó, h_status = 'fill'. Trả frame_log mới."""
        homs = {f: h[0] for f, h in geo["homs"].items()}
        filled = self.calib.fill_gaps(homs, geo["motion"], geo["shot_of"], geo["shape"]) if geo["shape"] else {}
        geo["n_filled"] = len(filled)
        if not filled:
            return frame_log
        for f, H in filled.items():
            geo["homs"][f] = (H, "fill", geo["homs"][f][2])
            geo["status"]["none"] -= 1
            geo["status"]["fill"] += 1
        by_f = defaultdict(list)
        for k, r in enumerate(frame_log):
            if r[0] in filled:
                by_f[r[0]].append(k)
        out = list(frame_log)
        for f, ks in by_f.items():
            boxes = np.array([frame_log[k][2:6] for k in ks], dtype=np.float32)
            pos = self._project_feet(filled[f], boxes, geo["shape"], geo, count=False)
            for k, p in zip(ks, pos):
                r = frame_log[k]
                out[k] = (*r[:9], None if p is None else round(float(p[0]), 2),
                          None if p is None else round(float(p[1]), 2), "fill")
        return out

    def _project_feet(self, H, boxes, shape, geo, count=True):
        """Điểm chân (đáy-giữa bbox) -> toạ độ sân (m). None nếu không chiếu được / không tin được.
        count=False: chiếu lại (fill_gaps) -> không đếm thêm vào tổng số cầu thủ."""
        pos = [None] * len(boxes)
        if count:
            geo["players"] += len(boxes)
        if H is None or not len(boxes):
            return pos
        L, W, m = self.pitch_cfg.length, self.pitch_cfg.width, config.PITCH_OUTSIDE_M
        feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
        xy, w = project(H, feet)
        for k, ((x, y), ww) in enumerate(zip(xy, w)):
            if boxes[k, 3] >= shape[0] - 2:            # chân bị cắt ở mép dưới frame -> điểm chân sai
                geo["cut_off"] += 1
            elif ww <= 0 or not np.isfinite([x, y]).all() or not (-m <= x <= L + m and -m <= y <= W + m):
                geo["outside"] += 1
            else:
                pos[k] = (x, y)
                geo["projected"] += 1
        return pos

    # ------------------------------------------------------------------ pass 2
    def render(self, src, dst, frame_log, numbers, roles, fps, size, prof: StepTimer, positions=None, geo=None,
               team_of=None, palette=None):
        """palette: đội -> màu BGR (màu áo thật của đội). Track có đội: bbox màu đội, viền dày nếu đã có số."""
        a = self.args
        by_frame = defaultdict(list)
        for row in frame_log:
            by_frame[row[0]].append(row)
        team_of, palette = team_of or {}, palette or {}
        det_num, det_src = numbers                    # (frame, track) -> số áo, nguồn ('read' | 'linked' | '')
        mm = MinimapRenderer(self.pitch_cfg, config.MINIMAP_WIDTH, palette=palette) if positions is not None else None
        tmp = dst.with_suffix(".tmp.mp4")
        t0 = time.perf_counter()
        n = 0
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        mm_tmp = dst.with_name(f"{src.stem}_minimap.tmp.mp4")
        mm_writer = (cv2.VideoWriter(str(mm_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, mm.size)
                     if mm is not None and a.minimap_video else None)
        last_dots, stale_n = [], 0
        for idx, frame in tqdm(read_frames(src, a.max_frames), total=None, desc="render"):
            rows = by_frame.get(idx, [])
            H, status, dbg = (geo["homs"].get(idx) or (None, "none", None)) if geo else (None, "none", None)
            if a.pitch_debug and geo is not None:
                draw_debug(frame, self.pitch_cfg, H, status, dbg, a.pitch_kpt_conf)
            for _, tid, x1, y1, x2, y2, *_ in rows:
                num = det_num.get((idx, tid), -1)
                # * = số có được nhờ nối, chưa tự đọc; ? = chỉ có nhờ nối qua lần gặp đồng đội (có thể nhầm người)
                mark = {"linked": "*", "uncertain": "?"}.get(det_src.get((idx, tid)), "")
                team = team_of.get((idx, tid), -1)
                if team in palette:
                    color = palette[team]
                    text = f"{'AB'[team]} #{num}{mark}" if num > 0 else f"{'AB'[team]} id{tid}"
                elif roles.get(tid) == "goalkeeper":
                    color = (0, 215, 255)
                    text = f"GK #{num}{mark}" if num > 0 else "GK"
                else:
                    color = (0, 220, 0) if num > 0 else (160, 160, 160)
                    text = f"#{num}{mark}" if num > 0 else ("ref" if roles.get(tid) == "referee" else f"id{tid}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)    # viền đen để thấy cả áo trắng/đen
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_ink(color), 2)
            if mm is not None:
                dots = [(*positions[(idx, row[1])], roles.get(row[1], "player"), det_num.get(row[:2], -1),
                         team_of.get(row[:2], -1)) for row in rows if (idx, row[1]) in positions]
                if H is not None or dots:
                    last_dots, stale_n = dots, 0
                    view = mm.view_polygon(H, frame.shape) if H is not None else None
                    img = mm.draw(dots, view=view, label=f"H: {status}" if a.pitch_debug else None)
                else:                                   # close-up / khán đài / replay: giữ vị trí cuối, vẽ mờ
                    stale_n += 1
                    keep = last_dots if stale_n <= config.MINIMAP_STALE_FRAMES else []
                    img = mm.draw(keep, stale=True, label="no pitch view")
                overlay(frame, img, a.minimap_pos)
                if mm_writer is not None:
                    mm_writer.write(img)
            writer.write(frame)
            n += 1
        writer.release()
        if mm_writer is not None:
            mm_writer.release()
        prof.add("9. render video", (time.perf_counter() - t0) * 1000, n_items=n)
        out = [to_h264(tmp, dst, prof)]
        if mm_writer is not None:
            out.append(to_h264(mm_tmp, dst.with_name(f"{src.stem}_minimap.mp4"), prof))
        return out

    # ------------------------------------------------------------------ run
    def process_video(self, src: Path, out_root: Path):
        a = self.args
        out_dir = out_root / src.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = None
        if a.save_crops:
            crops_dir = out_dir / "torso_crops"
            crops_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if a.max_frames:
            total = min(total or a.max_frames, a.max_frames)

        prof = StepTimer(self.device)
        t_start = time.perf_counter()
        voter, frame_log, span, stats, geo, teamer = self.analyse(read_frames(src, a.max_frames), total, prof, crops_dir)
        elapsed = time.perf_counter() - t_start

        t_agg = time.perf_counter()
        cls_votes = defaultdict(lambda: defaultdict(int))     # class của track = class xuất hiện nhiều nhất
        for row in frame_log:
            cls_votes[row[1]][row[7]] += 1
        roles = {t: role_of(self.class_names[max(v, key=v.get)]) for t, v in cls_votes.items()}
        team_of, team_info = {}, None                 # (frame, track) -> đội 0 | 1 | -1
        if teamer is not None:
            t_team = time.perf_counter()
            team_info = teamer.fit(roles)
            if team_info["ok"]:
                for t in teamer.as_player:            # detector gọi là goalkeeper nhưng áo đúng màu 1 đội
                    roles[t] = "player"
                team_of = {(r[0], r[1]): teamer.team_at(r[1], r[0]) for r in frame_log}
            prof.add("7c. team clustering", (time.perf_counter() - t_team) * 1000, n_items=len(roles))
        tracks, numbers = {}, {}
        for tid in sorted(span):
            num, weight = voter.best(tid)
            numbers[tid] = num
            tracks[tid] = {
                "role": roles.get(tid, "player"),
                **(teamer.track_summary(tid) if team_of else {}),
                "jersey_number": num,
                "weight": round(weight, 3),
                "votes": {str(k): round(v, 3) for k, v in sorted(voter.weights(tid).items(), key=lambda kv: -kv[1])},
                "first_frame": span[tid][0],
                "last_frame": span[tid][1],
                **stats[tid],
                "str_reads": len(voter.reads.get(tid, [])),
            }
        prof.add("7. aggregate (vote)", (time.perf_counter() - t_agg) * 1000, n_items=len(span))

        positions, mm_summary = None, None
        if a.minimap:
            with prof("7a. fill homography gaps", n_items=len(geo["homs"])):
                frame_log = self._fill_gaps(frame_log, geo)
            with prof("7b. smooth positions", n_items=len(frame_log)):
                positions = smooth_positions([(r[0], (r[1], r[8]), r[9], r[10]) for r in frame_log])
            mm_summary = minimap_summary(geo)

        # số áo theo từng detection: mặc định = số của cả track; với --link = số của cầu thủ (chuỗi nối)
        det_num = {(r[0], r[1]): numbers.get(r[1], -1) for r in frame_log}
        det_src = {k: ("read" if n > 0 else "") for k, n in det_num.items()}
        det_pid, players, link_stats = {}, None, None
        if positions is not None and a.link:
            with prof("7d. link identities", n_items=len(frame_log)):
                linker = IdentityLinker(fps, config, dict(filter_threshold=config.FILTER_THRESHOLD,
                                                          sum_threshold=a.sum_th, use_bias=config.USE_BIAS))
                linker.enc_split = a.enc_split
                linker.solver = a.link_solver
                ident, players, link_stats = linker.run([(r[0], r[1]) for r in frame_log], positions, team_of,
                                                        roles, geo["reads"], emb=geo["emb"] or None)
            for k, (pid, num, how) in ident.items():
                det_pid[k], det_num[k], det_src[k] = pid, num, how
            ids = defaultdict(lambda: defaultdict(int))     # track -> player_id -> số frame
            for (fr, tid), pid in det_pid.items():
                ids[tid][pid] += 1
            for tid, tr in tracks.items():
                tr["player_ids"] = dict(ids.get(tid, {}))

        t_out = time.perf_counter()
        with open(out_dir / "tracks.json", "w") as f:
            json.dump({"source": str(src), "fps": fps, "frame_size": list(size), "shots": geo["shots"],
                       "minimap": mm_summary,
                       "teams": team_info, "link": link_stats, "players": players, "tracks": tracks}, f, indent=2)
        with open(out_dir / "reads.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "label", "prob"])
            w.writerows((fr, t, lb, f"{p:.3f}") for fr, t, lb, p in geo["reads"])
        with open(out_dir / "detections.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "det_conf", "jersey_number", "number_source",
                        "player_id", "class", "team", "shot_id", "x_m", "y_m", "h_status"])
            for row in frame_log:
                k = row[:2]
                w.writerow([*row[:6], f"{row[6]:.3f}", det_num[k], det_src[k], det_pid.get(k, ""),
                            self.class_names[row[7]], team_of.get(k, -1), row[8],
                            "" if row[9] is None else row[9], "" if row[10] is None else row[10], row[11]])
        if positions is not None:
            # vị trí đã làm mượt, chỉ những frame chiếu được; toạ độ (0,0) = góc sân, x dọc chiều dài
            status = {f: h[1] for f, h in geo["homs"].items()}
            with open(out_dir / "positions.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "track_id", "player_id", "role", "team", "jersey_number", "number_source",
                            "shot_id", "h_status", "x_m", "y_m"])
                for row in frame_log:
                    p = positions.get((row[0], row[1]))
                    if p is not None:
                        k = row[:2]
                        w.writerow([row[0], row[1], det_pid.get(k, ""), roles.get(row[1], "player"),
                                    team_of.get(k, -1), det_num[k], det_src[k], row[8], status.get(row[0], ""),
                                    f"{p[0]:.2f}", f"{p[1]:.2f}"])
        prof.add("8. write json/csv", (time.perf_counter() - t_out) * 1000)

        n_frames = len(prof.calls.get("2. detect+track", []))
        identified = {t: n for t, n in numbers.items() if n > 0}
        print(f"\n[{src.name}] {n_frames} frames, {elapsed:.1f}s ({n_frames / max(elapsed, 1e-6):.1f} fps)")
        print(f"  tracks: {len(tracks)}, có số áo: {len(identified)}")
        if team_info is not None:
            if team_info["ok"]:
                for d in team_info["teams"]:
                    print(f"  đội {'AB'[d['team']]}: {d['n_tracks']} track cầu thủ, màu {d['dominant']} "
                          f"(BGR {d['color_bgr']})")
                det_p = [k for (f, t), k in team_of.items() if roles.get(t) == "player"]
                n_sw = sum(1 for t in tracks if tracks[t].get("team_switches", 0) > 0)
                print(f"    detection cầu thủ chưa rõ đội: {100 * det_p.count(-1) / max(len(det_p), 1):.1f}% | "
                      f"track đổi đội giữa chừng (đổi ID): {n_sw} | track 'goalkeeper' thực ra là cầu thủ "
                      f"(áo khớp màu đội): {team_info['n_goalkeeper_as_player']}")
            else:
                print(f"  chia đội: bỏ qua ({team_info['reason']})")
        for t, n in sorted(identified.items(), key=lambda kv: kv[1]):
            print(f"    track {t:>4}: #{n:<3} (weight {tracks[t]['weight']}, frames {span[t][0]}-{span[t][1]})")

        if link_stats is not None:
            ls = link_stats
            n_before = sum(1 for k, n in det_src.items() if n == "read")
            pl = sorted(((p["team"], p["number"], pid) for pid, p in players.items() if p["number"] > 0))
            print(f"  định danh (minimap): {ls['segments']} đoạn -> nối {ls['links']} lần "
                  f"(ghép {ls['solver']}; bỏ {ls['skipped_ambiguous']} cặp mập mờ, {ls['skipped_crowd']} cặp đám đông, "
                  f"{ls['skipped_number_conflict']} cặp khác số áo)")
            if a.reid:
                print(f"    re-ID: {ls['crowd_resolved_by_reid']} cặp đám đông được ngoại hình phân định -> "
                      f"nối {ls['links_reid']} lần (số áo qua đó = 'uncertain')")
            if a.enc_split:
                print(f"    gặp đồng đội: {ls['encounters']} lần -> nối qua lần gặp {ls['encounter_links']}, "
                      f"đoạn E gắn lại {ls['encounter_segments_attached']}/{ls['encounter_segments']}")
            print(f"    detection có số áo: tự đọc {100 * ls['dets_read'] / max(ls['dets_total'], 1):.1f}% + "
                  f"nhờ nối {100 * ls['dets_linked'] / max(ls['dets_total'], 1):.1f}% + "
                  f"không chắc {100 * ls['dets_uncertain'] / max(ls['dets_total'], 1):.1f}% | "
                  f"{ls['players_with_number']} cầu thủ có số: {', '.join(pid for _, _, pid in pl)}")
            print(f"    cảnh báo trùng (cùng player_id 2 lần trong 1 frame): {ls['dup_frame_ids']} lần")
        if mm_summary is not None:
            s = mm_summary
            print(f"  minimap: {geo['shots']} shot | frame có phép chiếu: kp {s['pct_kp']}%, flow {s['pct_flow']}%, "
                  f"lan ngược {s['pct_fill']}%, không có {s['pct_none']}% | lỗi reprojection trung vị "
                  + (f"{s['median_err_px']} px" if s['median_err_px'] is not None else f"{s['median_err_m']} m"))
            print(f"           cầu thủ chiếu lên sân được: {s['pct_projected']}% "
                  f"(ngoài sân -> bỏ {s['pct_outside']}%, chân bị cắt ở mép frame {s['pct_cut_off']}%)")
        elif geo["shots"] > 1:
            print(f"  shots: {geo['shots']} (tracker reset ở mỗi cắt cảnh)")
        if a.tile:
            print(f"  tile: thêm {geo['tile_extra']} box cầu thủ nhỏ mà full frame không thấy (trước tracker)")

        if not a.no_video:
            palette = ({k: tuple(d["color_bgr"]) for k, d in enumerate(team_info["teams"])}
                       if team_info and team_info["ok"] else {})
            paths = self.render(src, out_dir / f"{src.stem}_jersey.mp4", frame_log, (det_num, det_src), roles, fps, size,
                                prof, positions, geo if a.minimap else None, team_of, palette)
            for path in paths:
                print(f"  video: {path}")
        print(f"  json:  {out_dir / 'tracks.json'}")
        if positions is not None:
            print(f"  positions: {out_dir / 'positions.csv'}")

        timing = prof.summary(n_frames, time.perf_counter() - t_start)
        timing["analyse_fps"] = round(n_frames / max(elapsed, 1e-9), 2)
        timing["config"] = {"det_model": a.det_model, "torso_mode": a.torso_mode,
                            "pose_model": None if (self.det_is_pose or a.torso_mode == "bbox") else a.pose_model,
                            "imgsz": a.imgsz, "stride": a.stride, "no_legibility": a.no_legibility,
                            "minimap": a.minimap, "pitch_every": a.pitch_every if a.minimap else None,
                            "reset_on_cut": a.reset_on_cut, "teams": a.teams, "tracker": Path(a.tracker).name,
                            "conf": a.conf, "tile": a.tile, "enc_split": a.enc_split,
                            "link_solver": a.link_solver, "reid": a.reid}
        StepTimer.print_table(timing)
        print(f"  analyse (pass 1, không tính render): {timing['analyse_fps']} fps")
        with open(out_dir / "timing.json", "w") as f:
            json.dump(timing, f, indent=2, ensure_ascii=False)
        print(f"  timing: {out_dir / 'timing.json'}")


def minimap_summary(geo) -> dict:
    n = max(sum(geo["status"].values()), 1)
    p = max(geo["players"], 1)
    pct = lambda v, d: round(100 * v / d, 1)  # noqa: E731
    return {"shots": geo["shots"],
            "pct_kp": pct(geo["status"]["kp"], n), "pct_flow": pct(geo["status"]["flow"], n),
            "pct_fill": pct(geo["status"]["fill"], n), "pct_none": pct(geo["status"]["none"], n),
            "median_err_m": round(float(np.median(geo["err_m"])), 3) if geo["err_m"] else None,
            "median_err_px": round(float(np.median(geo["err_px"])), 2) if geo["err_px"] else None,
            "pct_projected": pct(geo["projected"], p), "pct_outside": pct(geo["outside"], p),
            "pct_cut_off": pct(geo["cut_off"], p)}


def to_h264(tmp: Path, dst: Path, prof: StepTimer) -> Path:
    """mp4v không xem được trong trình duyệt/notebook -> chuyển sang H.264 nếu có ffmpeg."""
    if shutil.which("ffmpeg"):
        with prof("10. ffmpeg -> H.264", n_items=1):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-preset", "fast", str(dst)], check=False)
        if dst.exists():
            tmp.unlink()
            return dst
    tmp.rename(dst)
    return dst


def resolve_classes(names: dict, spec) -> list:
    """Đổi --classes (id hoặc tên) thành list id. Không truyền -> tự chọn theo tên class.

    Tránh lỗi dùng id cứng: COCO có 0 = person, nhưng model bóng đá tự train
    thường có 0 = ball -> chỉ track được bóng.
    """
    lower = {str(n).lower(): i for i, n in names.items()}
    if not spec:
        ids = [i for n, i in lower.items() if n in config.PLAYER_CLASS_NAMES]
        if not ids:
            raise SystemExit(f"Không tìm thấy class cầu thủ trong detector {names}. "
                             f"Hãy chỉ định --classes (id hoặc tên).")
        return sorted(ids)
    ids = []
    for s in spec:
        if str(s).isdigit():
            i = int(s)
        elif str(s).lower() in lower:
            i = lower[str(s).lower()]
        else:
            raise SystemExit(f"--classes: không có class '{s}' trong detector {names}")
        if i not in names:
            raise SystemExit(f"--classes: id {i} không tồn tại trong detector {names}")
        ids.append(i)
    if any(str(names[i]).lower() in {"ball", "sports ball"} for i in ids):
        print("[warn] --classes đang gồm class bóng: bóng sẽ được track như cầu thủ")
    return sorted(set(ids))


def timed_frames(frames, prof: StepTimer):
    """Bọc generator đọc frame để đo thời gian decode video."""
    it = iter(frames)
    while True:
        t0 = time.perf_counter()
        try:
            item = next(it)
        except StopIteration:
            return
        prof.add("1. decode frame", (time.perf_counter() - t0) * 1000)
        yield item


def read_frames(src: Path, max_frames: int = 0):
    """Yield (frame_idx, frame BGR). Ảnh đơn được coi là video 1 frame."""
    if src.suffix.lower() in IMAGE_EXTS:
        yield 0, cv2.imread(str(src))
        return
    cap = cv2.VideoCapture(str(src))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        yield idx, frame
        idx += 1
    cap.release()


def collect_sources(source: str):
    p = Path(source)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
    else:
        files = [p]
    if not files:
        raise SystemExit(f"Không tìm thấy video/ảnh trong {source}")
    return files


def main():
    args = parse_args()
    pipe = PlayerJerseyPipeline(args)
    out_root = Path(args.output)
    for src in collect_sources(args.source):
        pipe.process_video(src, out_root)


if __name__ == "__main__":
    main()
