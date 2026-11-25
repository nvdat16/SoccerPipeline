"""Re-ID theo ngoại hình: embedding OSNet-AIN (512 chiều, chuẩn hoá L2) cho crop cầu thủ.

Dùng trong IdentityLinker (--reid) để phân biệt 2 cầu thủ CÙNG ĐỘI khi vị trí / chuyển động trên sân không đủ
(đám đông, mập mờ). Màu áo giống hệt nhau nên embedding chủ yếu dựa vào dáng người, tóc, da, tất, giày, số áo.

Kiến trúc OSNet-AIN rút gọn từ deep-person-reid / torchreid (Kaiyang Zhou, MIT License):
  Zhou et al. "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019.
  Zhou et al. "Learning Generalisable Omni-Scale Representations for Person Re-Identification", TPAMI 2021.
Weights osnet_ain_x1_0 train trên MSMT17 (model zoo của torchreid). Chỉ giữ phần inference.
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------- layers (giữ tên module để load weights)
class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(cout, affine=True) if IN else nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, cin, cout, depth):
        super().__init__()
        self.layers = nn.Sequential(LightConv3x3(cin, cout), *[LightConv3x3(cout, cout) for _ in range(depth - 1)])

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, cin, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cin, cin // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(cin // reduction, cin, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    def __init__(self, cin, cout, reduction=4, T=4, IN=False):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout, bn=not IN)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
        if IN:
            self.IN = nn.InstanceNorm2d(cout, affine=True)
        self.use_in = IN

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2 = x2 + self.gate(conv2_t(x1))
        x3 = self.conv3(x2)
        if self.use_in:
            x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSBlockINin(OSBlock):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, IN=True)


class OSNetAIN(nn.Module):
    """osnet_ain_x1_0: channels 64-256-384-512, feature 512."""

    def __init__(self, channels=(64, 256, 384, 512), feature_dim=512):
        super().__init__()
        blocks = [[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]]
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(blocks[0][0](c[0], c[1]), blocks[0][1](c[1], c[1]))
        self.pool2 = nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(blocks[1][0](c[1], c[2]), blocks[1][1](c[2], c[2]))
        self.pool3 = nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = nn.Sequential(blocks[2][0](c[2], c[3]), blocks[2][1](c[3], c[3]))
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())

    def forward(self, x):
        x = self.conv5(self.conv4(self.pool3(self.conv3(self.pool2(self.conv2(self.maxpool(self.conv1(x))))))))
        return self.fc(self.global_avgpool(x).flatten(1))


# ---------------------------------------------------------------------------- OSNet x1.0 thường (osnet.py của torchreid)
class OSBlockV1(nn.Module):
    """OSBlock của osnet.py: 4 nhánh tên conv2a..conv2d (khác ModuleList conv2 của bản AIN), không IN."""

    def __init__(self, cin, cout, reduction=4):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = sum(self.gate(s(x1)) for s in (self.conv2a, self.conv2b, self.conv2c, self.conv2d))
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.conv3(x2) + identity)


class OSNetX1(nn.Module):
    """osnet_x1_0: channels 64-256-384-512, feature 512. Pool nằm TRONG conv2/conv3 (index 2) như torchreid."""

    def __init__(self, channels=(64, 256, 384, 512), feature_dim=512):
        super().__init__()
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(OSBlockV1(c[0], c[1]), OSBlockV1(c[1], c[1]),
                                   nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2)))
        self.conv3 = nn.Sequential(OSBlockV1(c[1], c[2]), OSBlockV1(c[2], c[2]),
                                   nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2)))
        self.conv4 = nn.Sequential(OSBlockV1(c[2], c[3]), OSBlockV1(c[3], c[3]))
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        return self.fc(self.global_avgpool(x).flatten(1))


# ---------------------------------------------------------------------------- wrapper
class OSNetEmbedder:
    SIZE = (128, 256)                                     # (w, h) như lúc train
    MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
    STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

    def __init__(self, weights, device, batch=64):
        # checkpoint torchreid gốc (vd Deep-EIoU sports_model.pth.tar-60) có numpy scalar (rank1) -> cho phép riêng
        # mấy class numpy vô hại đó, vẫn weights_only=True (không chạy code tuỳ ý trong pickle)
        try:
            from numpy._core.multiarray import scalar                     # numpy >= 2
        except ImportError:
            from numpy.core.multiarray import scalar                      # numpy 1.x
        safe = [(scalar, "numpy.core.multiarray.scalar"), np.dtype] + \
               [type(np.dtype(t)) for t in ("float32", "float64", "int64")]
        with torch.serialization.safe_globals(safe):
            sd = torch.load(str(weights), map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items() if "classifier" not in k}
        self.arch = "osnet_x1_0" if any(".conv2a." in k for k in sd) else "osnet_ain_x1_0"   # tự nhận theo tên key
        self.model = OSNetX1() if self.arch == "osnet_x1_0" else OSNetAIN()
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"OSNet weights không khớp: thiếu {missing[:5]}, thừa {unexpected[:5]}")
        self.device, self.batch = device, batch
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, crops):
        """crops: list ảnh BGR -> np.float16 (N, 512) đã chuẩn hoá L2."""
        out = []
        for i in range(0, len(crops), self.batch):
            x = np.stack([(cv2.resize(c, self.SIZE, interpolation=cv2.INTER_LINEAR)[:, :, ::-1] - self.MEAN)
                          / self.STD for c in crops[i:i + self.batch]]).astype(np.float32)
            f = self.model(torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device))
            out.append(F.normalize(f, dim=1).cpu().numpy().astype(np.float16))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float16)
