"""Nhận dạng số áo bằng PARSeq (checkpoint fine-tune trên SoccerNet).

Dùng lại code `strhub` trong jersey-number-pipeline/str/parseq. Logic decode và
tính confidence giống `run_inference` trong jersey-number-pipeline/str.py.
"""
import string
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

import config

if str(config.PARSEQ_DIR) not in sys.path:
    sys.path.insert(0, str(config.PARSEQ_DIR))

from strhub.models.parseq.system import PARSeq  # noqa: E402


def load_parseq(checkpoint) -> PARSeq:
    """Tương đương PARSeq.load_from_checkpoint nhưng tương thích torch>=2.6
    (checkpoint PL 1.9 chứa object Python nên cần weights_only=False)."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    hparams["charset_test"] = string.digits
    model = PARSeq(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    return model


class JerseyNumberRecognizer:
    def __init__(self, checkpoint, device: str):
        self.device = device
        self.model = load_parseq(checkpoint).eval().to(device)
        img_size = self.model.hparams.img_size
        # = SceneTextDataModule.get_transform(img_size) (viết lại để khỏi phụ thuộc lmdb)
        self.transform = T.Compose([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    @torch.no_grad()
    def predict(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Trả về [(label, prob)] cho từng torso crop. prob = tích xác suất các ký tự (bỏ EOS)."""
        if not crops_bgr:
            return []
        batch = torch.stack([
            self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr
        ]).to(self.device)
        logits = self.model(batch)
        # tối đa 2 chữ số + EOS, 11 token đầu = EOS + '0'..'9'
        probs_full = logits[:, :3, :11].softmax(-1)
        preds, probs = self.model.tokenizer.decode(probs_full)
        out = []
        for label, p in zip(preds, probs):
            p = p.float().cpu()
            out.append((label, float(p[:-1].prod()) if len(p) > 1 else 0.0))
        return out
