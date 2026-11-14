"""Legibility classifier (ResNet34) — ảnh cầu thủ có nhìn thấy/đọc được số áo không.

Kiến trúc và tiền xử lý giống `LegibilityClassifier34` + transform 'test' trong
jersey-number-pipeline, để load trực tiếp weights SoccerNet đã train sẵn.
"""
from typing import List

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


class LegibilityClassifier34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet34(weights=None)
        self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.model_ft(x))


class LegibilityModel:
    def __init__(self, weights_path, device: str, threshold: float = 0.5):
        self.device = device
        self.threshold = threshold
        self.model = LegibilityClassifier34()
        state = torch.load(str(weights_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Trả về xác suất legible cho từng crop (BGR)."""
        if not crops_bgr:
            return np.zeros(0)
        batch = torch.stack([
            self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr
        ]).to(self.device)
        return self.model(batch).squeeze(1).float().cpu().numpy()
