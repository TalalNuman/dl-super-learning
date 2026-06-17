"""
MSDS25011 - Assignment 5, Task 3: Feature Similarity Before Training
Measure average cosine similarity between same-image and different-image
view pairs using a random untrained encoder.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from MSDS25011_05_task2_augmentations import TwoViewTransform, simclr_transform
from MSDS25011_05_task4_simclr import SimCLRModel, load_split

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_avg_similarities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute average cosine similarity for same-image view pairs (diagonal)
    and different-image view pairs (off-diagonal) using encoder features.
    """
    model.eval()
    total_same_sim = 0.0
    total_diff_sim = 0.0
    total_same_count = 0
    total_diff_count = 0

    for (x_i, x_j), _ in dataloader:
        x_i, x_j = x_i.to(device), x_j.to(device)

        h_i = model.get_features(x_i)
        h_j = model.get_features(x_j)

        h_i_norm = torch.nn.functional.normalize(h_i, p=2, dim=1)
        h_j_norm = torch.nn.functional.normalize(h_j, p=2, dim=1)

        sim_matrix = torch.matmul(h_i_norm, h_j_norm.T)

        same_sims = torch.diag(sim_matrix)
        total_same_sim += same_sims.sum().item()
        total_same_count += same_sims.numel()

        n = sim_matrix.shape[0]
        if n > 1:
            mask = ~torch.eye(n, dtype=torch.bool, device=device)
            diff_sims = sim_matrix[mask]
            total_diff_sim += diff_sims.sum().item()
            total_diff_count += diff_sims.numel()

    avg_same = total_same_sim / total_same_count if total_same_count > 0 else 0.0
    avg_diff = total_diff_sim / total_diff_count if total_diff_count > 0 else 0.0
    return avg_same, avg_diff


if __name__ == '__main__':
    BATCH_SIZE = 64

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}\n')

    two_view_tf = TwoViewTransform(simclr_transform)
    val_dataset = load_split('splits/val.txt', transform=two_view_tf)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = SimCLRModel().to(device)
    print('Computing average same/different image similarities (Before Training)...')
    avg_same, avg_diff = compute_avg_similarities(model, val_loader, device)

    print(f'  Average Same-Image Similarity     : {avg_same:.4f}')
    print(f'  Average Different-Image Similarity: {avg_diff:.4f}')
