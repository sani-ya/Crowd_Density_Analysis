"""
model.py — Crowd counting and forecasting models.

Architectures provided:
  1. CSRNet  — VGG-16 frontend + dilated convolution backend.
               Best accuracy on ShanghaiTech / UCF benchmarks.
  2. MCNN    — Multi-column CNN; handles extreme density variation.
  3. CrowdForecaster — GRU/LSTM for 5-10 minute crowd prediction.

──────────────────────────────────────────────────────────────────
Architecture Rationale
──────────────────────────────────────────────────────────────────
WHY CSRNet?
  • VGG-16 frontend (pretrained on ImageNet) extracts rich multi-scale features.
  • Dilated convolutions in the backend expand receptive field WITHOUT
    downsampling — critical for dense, occluded crowd scenes.
  • Output stride = 8 produces high-res density maps (1/8 of input).
  • State-of-the-art on ShanghaiTech Part A & B, UCF_CC_50.
  Reference: Li et al. "CSRNet: Dilated Convolutional Neural Networks for
             Understanding the Highly Congested Scenes." CVPR 2018.

WHY MCNN?
  • Three parallel columns with different filter sizes (9×9, 7×7, 5×5).
  • Each column specialises in crowds at different scales/densities.
  • No pretrained backbone — trains from scratch, lighter.
  Reference: Zhang et al. "Single-Image Crowd Counting via Multi-Column CNN." CVPR 2016.

TRANSFORMER IMPROVEMENT SUGGESTION:
  Replace or augment the VGG backend with a Swin Transformer or ViT backbone:
  • Swin Transformer provides hierarchical features with shifted windows.
  • Cross-attention between spatial tokens naturally models crowd co-occurrence.
  • CrowdFormer (ICCV 2021) and GL (IEEE TIP 2022) show Transformers
    outperform CSRNet by 15-20% MAE on ShanghaiTech Part A.
  Implementation stub: see CrowdTransformerEncoder below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


# ──────────────────────────────────────────────────────────────
#  Helper: make_layers (VGG-style)
# ──────────────────────────────────────────────────────────────

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    Build a sequential block from a layer config list.
    'M' = MaxPool, integer = Conv2d out_channels.
    dilation=True uses dilated convolutions (rate=2) in final blocks.
    """
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            d = 2 if dilation else 1
            conv = nn.Conv2d(in_channels, v, kernel_size=3,
                             padding=d, dilation=d)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────
#  CSRNet
# ──────────────────────────────────────────────────────────────

class CSRNet(nn.Module):
    """
    CSRNet: Dilated CNN for crowd density estimation.

    Frontend: VGG-16 layers 1–13 (before pool4), stride=8 total.
    Backend: 6 dilated conv layers (dilation=2), then 1×1 → density map.

    Input:  (B, 3, H, W)
    Output: (B, 1, H/8, W/8)  — density map
    """

    # VGG-16 first 10 blocks (up to conv4_3, before pool4)
    FRONTEND_CONFIG = [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'M',
        512, 512, 512,    # no pool4 — preserve spatial resolution
    ]

    BACKEND_CONFIG = [512, 512, 512, 256, 128, 64]

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        self.frontend = make_layers(self.FRONTEND_CONFIG)
        self.backend  = make_layers(self.BACKEND_CONFIG,
                                    in_channels=512, dilation=True)
        self.output   = nn.Conv2d(64, 1, kernel_size=1)
        self.dropout  = nn.Dropout2d(p=dropout)

        # Initialise backend weights
        self._init_weights(self.backend)
        self._init_weights(self.output)

        # Load VGG-16 pretrained weights into frontend
        if pretrained:
            self._load_pretrained_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.dropout(x)
        x = self.backend(x)
        x = self.output(x)
        return x   # (B, 1, H/8, W/8)

    def _load_pretrained_frontend(self):
        """Copy VGG-16 ImageNet weights to frontend layers."""
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # VGG-16 features[:23] = first 13 conv layers
        pretrained_state = list(vgg.features[:23].state_dict().items())
        our_state = list(self.frontend.state_dict().items())
        # Only copy matching weight shapes
        for (our_k, our_v), (pre_k, pre_v) in zip(our_state, pretrained_state):
            if our_v.shape == pre_v.shape:
                our_v.copy_(pre_v)

    @staticmethod
    def _init_weights(layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ──────────────────────────────────────────────────────────────
#  MCNN  (Multi-Column CNN)
# ──────────────────────────────────────────────────────────────

class MCNNColumn(nn.Module):
    """Single column of MCNN for one scale."""

    def __init__(self, filter_size: int):
        super().__init__()
        p = filter_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(1,  16, filter_size, padding=p), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16,  8, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MCNN(nn.Module):
    """
    Multi-Column CNN for crowd counting.

    Input:  (B, 3, H, W) — converts to grayscale internally.
    Output: (B, 1, H/4, W/4)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.col1 = MCNNColumn(9)
        self.col2 = MCNNColumn(7)
        self.col3 = MCNNColumn(5)
        self.merge = nn.Conv2d(24, 1, kernel_size=1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB → grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        out1 = self.col1(gray)
        out2 = self.col2(gray)
        out3 = self.col3(gray)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.merge(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ──────────────────────────────────────────────────────────────
#  Crowd Forecaster (GRU / LSTM)
# ──────────────────────────────────────────────────────────────

class CrowdForecaster(nn.Module):
    """
    Temporal model for short-term crowd prediction (5–10 minutes).

    Architecture:
      Linear projection → GRU/LSTM encoder → Linear decoder → predictions

    Input:  (B, seq_len, input_size)   — e.g. (B, 30, 1) for 30 past counts
    Output: (B, forecast_horizon)      — next N crowd counts
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 10,
        rnn_type: str = 'GRU',
        dropout: float = 0.2,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size

        RNN = nn.GRU if rnn_type.upper() == 'GRU' else nn.LSTM
        self.rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, input_size)
        returns: (B, forecast_horizon)
        """
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]   # take final timestep
        return self.fc(last_hidden)


# ──────────────────────────────────────────────────────────────
#  Transformer Stub (Research Extension)
# ──────────────────────────────────────────────────────────────

class CrowdTransformerEncoder(nn.Module):
    """
    STUB — Transformer-based backbone for crowd counting.

    Replaces CSRNet's VGG frontend with a Swin Transformer encoder,
    providing hierarchical attention-based feature extraction.

    To use (requires timm library):
        pip install timm
        from model import CrowdTransformerEncoder
        model = CrowdTransformerEncoder()

    Performance advantage:
        ShanghaiTech Part A MAE: CSRNet ~68 → Swin-based ~54 (≈21% improvement)
    """

    def __init__(self):
        super().__init__()
        # Uncomment once timm is installed:
        # import timm
        # self.backbone = timm.create_model('swin_tiny_patch4_window7_224',
        #                                   pretrained=True, features_only=True)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(768, 256, 2, 2),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 1, 1),
        # )
        raise NotImplementedError("Install timm and uncomment backbone code.")

    def forward(self, x):
        feats = self.backbone(x)
        return self.decoder(feats[-1])


# ──────────────────────────────────────────────────────────────
#  Model Factory
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'csrnet': CSRNet,
    'mcnn':   MCNN,
}


def build_model(cfg: dict) -> nn.Module:
    """Instantiate counting model from config."""
    arch = cfg['model']['architecture'].lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[arch]
    kwargs = {
        'pretrained': cfg['model'].get('pretrained_backbone', True),
        'dropout':    cfg['model'].get('dropout', 0.5),
    }
    return model_cls(**kwargs)


def build_forecaster(cfg: dict) -> CrowdForecaster:
    """Instantiate forecasting model from config."""
    fc = cfg['forecasting']
    return CrowdForecaster(
        input_size=1,
        hidden_size=fc['hidden_size'],
        num_layers=fc['num_layers'],
        forecast_horizon=fc['forecast_horizon'],
        rnn_type=fc['model'],
        dropout=fc['dropout'],
    )
