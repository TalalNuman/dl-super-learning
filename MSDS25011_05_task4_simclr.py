"""
Task 4: SimCLR Model
Defines the SimCLR architecture — a CIFAR-10-adapted ResNet-18 encoder
with a two-layer MLP projection head.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ── SimCLR Model ────────────────────────────────────────────────────────────
class SimCLRModel(nn.Module):
    """
    SimCLR framework built on a CIFAR-10-adapted ResNet-18.

    Architecture
    ------------
    encoder  : ResNet-18 (conv1 → 3×3, no maxpool, fc removed)  →  512-d
    projector: Linear(512, 256) → ReLU → Linear(256, 128)       →  128-d

    Methods
    -------
    forward(x)       → z  (128-d projection, used during pretraining)
    get_features(x)  → h  (512-d encoder output, used for downstream tasks)
    """

    def __init__(self, projection_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        # ── Encoder: ResNet-18 adapted for 32×32 inputs ────────────────
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        # Remove the original fc layer; keep everything else
        encoder_out_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.encoder = backbone

        # ── Projection head (MLP) ──────────────────────────────────────
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        """Return projected representation z (128-d)."""
        h = self.encoder(x)
        z = self.projector(h)
        return z

    def get_features(self, x):
        """Return encoder features h (512-d), bypassing the projection head."""
        return self.encoder(x)


# ── Main: print model summary ──────────────────────────────────────────────
if __name__ == '__main__':
    model = SimCLRModel()

    # ── Encoder summary ────────────────────────────────────────────────
    print('=' * 65)
    print('SimCLR Model Summary')
    print('=' * 65)

    print('\n── Encoder (CIFAR-10 ResNet-18) ──')
    print(model.encoder)

    print('\n── Projection Head ──')
    print(model.projector)

    # ── Parameter counts ───────────────────────────────────────────────
    encoder_params   = sum(p.numel() for p in model.encoder.parameters())
    projector_params = sum(p.numel() for p in model.projector.parameters())
    total_params     = sum(p.numel() for p in model.parameters())

    print('\n── Parameter Counts ──')
    print(f'  Encoder    : {encoder_params:>10,}')
    print(f'  Projector  : {projector_params:>10,}')
    print(f'  Total      : {total_params:>10,}')

    # ── Quick shape check ──────────────────────────────────────────────
    dummy = torch.randn(2, 3, 32, 32)
    z = model(dummy)
    h = model.get_features(dummy)
    print(f'\n── Shape Check (batch=2, 3×32×32 input) ──')
    print(f'  forward()       → z.shape = {tuple(z.shape)}   (projection)')
    print(f'  get_features()  → h.shape = {tuple(h.shape)}  (encoder)')
    print('=' * 65)
