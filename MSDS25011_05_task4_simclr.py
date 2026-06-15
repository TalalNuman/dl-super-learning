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


# ── Similarity Matrix and Loss Functions ────────────────────────────────────
def compute_similarity_matrix(z: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity matrix of shape (2N, 2N) for batch z (2N, D).
    Each row of z is L2-normalized first.
    """
    z_normalized = torch.nn.functional.normalize(z, p=2, dim=1)
    sim_matrix = torch.matmul(z_normalized, z_normalized.T)
    return sim_matrix


def nt_xent_loss(z: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Computes the NT-Xent loss for batch z of shape (2N, D).
    The first N rows are view1, and the next N rows are view2.
    """
    two_n = z.shape[0]
    n = two_n // 2
    
    # Cosine similarity matrix scaled by temperature
    sim_matrix = compute_similarity_matrix(z) / temperature
    
    # Mask to remove self-similarity (diagonal)
    mask = ~torch.eye(two_n, dtype=torch.bool, device=z.device)
    
    # Filter out diagonal and reshape to (2N, 2N - 1)
    logits = sim_matrix[mask].view(two_n, two_n - 1)
    
    # Target positive pairs index map:
    # For sample i (0 <= i < N), the positive is sample i + N.
    # Since we remove the diagonal (self-similarity at i), the index shifts:
    # - If i < N, positive index is i + N. Since i + N > i, shifts left by 1 to: i + N - 1.
    # - If i >= N, positive index is i - N. Since i - N < i, does not shift: i - N.
    targets = torch.arange(two_n, device=z.device)
    targets = torch.where(targets < n, targets + n - 1, targets - n)
    
    loss = torch.nn.functional.cross_entropy(logits, targets)
    return loss


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torchvision import datasets
    from MSDS25011_05_task2_augmentations import TwoViewTransform, simclr_transform

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

    # ── Loss and Heatmap Verification ──────────────────────────────────
    print('\nGenerating Similarity Matrix Heatmap (Before Training)...')
    os.makedirs('results', exist_ok=True)

    # Load 4 images and apply TwoViewTransform
    two_view_tf = TwoViewTransform(simclr_transform)
    cifar_two_view = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=two_view_tf
    )
    
    # Collate 4 samples (each is ((view1, view2), label))
    v1_list, v2_list = [], []
    for i in range(4):
        (v1, v2), _ = cifar_two_view[i]
        v1_list.append(v1)
        v2_list.append(v2)
        
    # Stack views so the first N are view1, next N are view2
    batch = torch.stack(v1_list + v2_list, dim=0) # shape (8, 3, 32, 32)
    
    # Pass through random model
    model.eval()
    with torch.no_grad():
        z_batch = model(batch)
        sim_matrix = compute_similarity_matrix(z_batch).cpu().numpy()
        loss_val = nt_xent_loss(z_batch, temperature=0.5).item()
        
    print(f'NT-Xent Loss on this batch: {loss_val:.4f}')

    # Plot Similarity Matrix Heatmap
    labels = [
        'Img1_V1', 'Img2_V1', 'Img3_V1', 'Img4_V1',
        'Img1_V2', 'Img2_V2', 'Img3_V2', 'Img4_V2'
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    # Loop over data dimensions and create text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                    ha="center", va="center", color="black" if abs(sim_matrix[i, j]) < 0.7 else "white")
            
    ax.set_title("Cosine Similarity Matrix Before Training\n(Untrained ResNet-18 + MLP projection head)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/similarity_matrix_before_training.png', dpi=150)
    plt.close()
    print('Pre-training similarity heatmap saved → results/similarity_matrix_before_training.png')

