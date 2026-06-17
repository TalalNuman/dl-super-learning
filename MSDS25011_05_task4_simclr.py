"""
Task 4: SimCLR Model
Defines the SimCLR architecture — a CIFAR-10-adapted ResNet-18 encoder
with a two-layer MLP projection head, cosine similarity, and NT-Xent loss.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models

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

        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        encoder_out_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.encoder = backbone

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

    sim_matrix = compute_similarity_matrix(z) / temperature
    mask = ~torch.eye(two_n, dtype=torch.bool, device=z.device)
    logits = sim_matrix[mask].view(two_n, two_n - 1)

    targets = torch.arange(two_n, device=z.device)
    targets = torch.where(targets < n, targets + n - 1, targets - n)

    loss = torch.nn.functional.cross_entropy(logits, targets)
    return loss


def load_split(split_file: str, transform, cifar_root: str = './data') -> torch.utils.data.Subset:
    """Load a CIFAR-10 subset defined by an index file with a custom transform."""
    is_train = 'test' not in os.path.basename(split_file)
    full_dataset = datasets.CIFAR10(
        root=cifar_root,
        train=is_train,
        download=True,
        transform=transform,
    )
    with open(split_file, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    return torch.utils.data.Subset(full_dataset, indices)


def save_similarity_heatmap(sim_matrix: np.ndarray, filename: str, title: str):
    """Saves an 8x8 similarity matrix heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [
        'Img1_V1', 'Img2_V1', 'Img3_V1', 'Img4_V1',
        'Img1_V2', 'Img2_V2', 'Img3_V2', 'Img4_V2',
    ]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va='bottom')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, f'{sim_matrix[i, j]:.2f}',
                ha='center', va='center',
                color='black' if abs(sim_matrix[i, j]) < 0.7 else 'white',
            )

    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def visualize_4_samples(
    model: nn.Module,
    dataset: torch.utils.data.Subset,
    device: torch.device,
    filename: str,
    title: str,
):
    """Extracts 4 samples, processes through the model, and plots heatmap."""
    v1_list, v2_list = [], []
    for i in range(4):
        (v1, v2), _ = dataset[i]
        v1_list.append(v1)
        v2_list.append(v2)

    batch = torch.stack(v1_list + v2_list, dim=0).to(device)
    model.eval()
    with torch.no_grad():
        z_batch = model(batch)
        sim_matrix = compute_similarity_matrix(z_batch).cpu().numpy()

    save_similarity_heatmap(sim_matrix, filename, title)


if __name__ == '__main__':
    from MSDS25011_05_task2_augmentations import TwoViewTransform, simclr_transform

    os.makedirs('results', exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}\n')

    two_view_tf = TwoViewTransform(simclr_transform)
    ssl_dataset = load_split('splits/train_ssl_unlabeled.txt', transform=two_view_tf)

    model = SimCLRModel().to(device)
    visualize_4_samples(
        model, ssl_dataset, device,
        'results/similarity_matrix_before_training.png',
        'Cosine Similarity Matrix Before Training\n(Untrained ResNet-18 + MLP head)',
    )
    print('Before-training heatmap saved → results/similarity_matrix_before_training.png')
