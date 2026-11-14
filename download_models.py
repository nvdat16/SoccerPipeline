"""Tải weights legibility + PARSeq (SoccerNet) từ Google Drive về MODELS_DIR.

YOLO weights (yolo11*-pose.pt) được ultralytics tự tải khi chạy lần đầu.
"""
import argparse
from pathlib import Path

import gdown

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(config.MODELS_DIR))
    args = ap.parse_args()

    out_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in config.MODEL_URLS.items():
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {dst} đã có")
            continue
        print(f"[download] {name}")
        gdown.download(url, str(dst), quiet=False)
        if not dst.exists():
            raise SystemExit(
                f"Tải {name} thất bại (Google Drive có thể bị giới hạn lượt tải). "
                f"Tải thủ công từ {url} rồi upload lên Kaggle Dataset và set MODELS_DIR."
            )
    print("Done.")


if __name__ == "__main__":
    main()
