"""
Task 4: SimCLR Model
Defines the SimCLR architecture — a CIFAR-10-adapted ResNet-18 encoder
with a two-layer MLP projection head.
"""

import json
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


@torch.no_grad()
def compute_avg_similarities(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Computes average cosine similarity for same-image views and different-image views
    using the model's encoder features (get_features).
    """
    model.eval()
    total_same_sim = 0.0
    total_diff_sim = 0.0
    total_same_count = 0
    total_diff_count = 0
    
    for (x_i, x_j), _ in dataloader:
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        # Get 512-dim features
        h_i = model.get_features(x_i)
        h_j = model.get_features(x_j)
        
        h_i_norm = torch.nn.functional.normalize(h_i, p=2, dim=1)
        h_j_norm = torch.nn.functional.normalize(h_j, p=2, dim=1)
        
        # Compute cosine similarity matrix between view1 and view2
        sim_matrix = torch.matmul(h_i_norm, h_j_norm.T)
        
        # Same-image similarity (diagonal of sim_matrix)
        same_sims = torch.diag(sim_matrix)
        total_same_sim += same_sims.sum().item()
        total_same_count += same_sims.numel()
        
        # Different-image similarity (off-diagonal elements of sim_matrix)
        n = sim_matrix.shape[0]
        if n > 1:
            mask = ~torch.eye(n, dtype=torch.bool, device=device)
            diff_sims = sim_matrix[mask]
            total_diff_sim += diff_sims.sum().item()
            total_diff_count += diff_sims.numel()
            
    avg_same = total_same_sim / total_same_count if total_same_count > 0 else 0.0
    avg_diff = total_diff_sim / total_diff_count if total_diff_count > 0 else 0.0
    return avg_same, avg_diff


def save_similarity_metrics(
    pre_same: float,
    pre_diff: float,
    post_same: float | None = None,
    post_diff: float | None = None,
    *,
    batch_size: int = 64,
    out_path: str = 'results/metrics.json',
) -> None:
    """Persist before/after similarity stats for the report and metrics.json."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    metrics: dict = {}
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            metrics = json.load(f)

    metrics.update({
        'seed': SEED,
        'batch_size': batch_size,
        'simclr_epochs': 50,
        'learning_rate': 0.0003,
        'temperature': 0.5,
        'same_view_similarity_before': round(pre_same, 6),
        'different_image_similarity_before': round(pre_diff, 6),
    })
    if post_same is not None and post_diff is not None:
        metrics['same_view_similarity_after'] = round(post_same, 6)
        metrics['different_image_similarity_after'] = round(post_diff, 6)

    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_split(split_file: str, transform, cifar_root: str = './data') -> torch.utils.data.Subset:
    """
    Load a CIFAR-10 subset defined by an index file with a custom transform.
    """
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
        'Img1_V2', 'Img2_V2', 'Img3_V2', 'Img4_V2'
    ]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")
    
    # Label axes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    # Annotate values
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                    ha="center", va="center", color="black" if abs(sim_matrix[i, j]) < 0.7 else "white")
            
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def visualize_4_samples(model: nn.Module, dataset: torch.utils.data.Subset, device: torch.device, filename: str, title: str):
    """Extracts 4 samples, processes through the model, and plots heatmap."""
    v1_list, v2_list = [], []
    for i in range(4):
        (v1, v2), _ = dataset[i]
        v1_list.append(v1)
        v2_list.append(v2)
        
    batch = torch.stack(v1_list + v2_list, dim=0).to(device) # shape (8, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        z_batch = model(batch)
        sim_matrix = compute_similarity_matrix(z_batch).cpu().numpy()
        
    save_similarity_heatmap(sim_matrix, filename, title)


# ── Main Pretraining Script ──────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from MSDS25011_05_task2_augmentations import TwoViewTransform, simclr_transform

    # Output directories
    for d in ('graphs', 'results', 'models'):
        os.makedirs(d, exist_ok=True)

    # ── Device selection ────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}\n')

    # ── Hyperparameters ─────────────────────────────────────────────────
    # Use 32 instead of 64 if MPS runs out of memory
    BATCH_SIZE  = 64
    LR          = 3e-4
    EPOCHS      = 50
    TEMPERATURE = 0.5
    SSL_SPLIT   = 'splits/train_ssl_unlabeled.txt'

    # ── 1. Load Data ────────────────────────────────────────────────────
    print('Loading unlabeled SSL subset...')
    two_view_tf = TwoViewTransform(simclr_transform)
    ssl_dataset = load_split(SSL_SPLIT, transform=two_view_tf)
    print(f'SSL pretraining samples: {len(ssl_dataset):,}\n')

    ssl_loader = torch.utils.data.DataLoader(
        ssl_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, drop_last=True
    )

    # ── 2. Initialize Model and Optimizer ──────────────────────────────
    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── 3. Compute Untrained Similarities ───────────────────────────────
    print('Computing average same/different image similarities (Before Training)...')
    # Use a small subset dataloader to speed up similarity evaluation
    sim_dataset = load_split('splits/val.txt', transform=two_view_tf)
    sim_loader = torch.utils.data.DataLoader(sim_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    pre_same, pre_diff = compute_avg_similarities(model, sim_loader, device)
    print(f'Before Training:')
    print(f'  Average Same-Image Similarity     : {pre_same:.4f}')
    print(f'  Average Different-Image Similarity: {pre_diff:.4f}\n')
    save_similarity_metrics(pre_same, pre_diff, batch_size=BATCH_SIZE)
    print('Before-training similarities saved → results/metrics.json\n')

    # Heatmap before training
    visualize_4_samples(
        model, ssl_dataset, device, 
        'results/similarity_matrix_before_training.png',
        "Cosine Similarity Matrix Before Training\n(Untrained ResNet-18 + MLP head)"
    )
    print('Before-training heatmap saved → results/similarity_matrix_before_training.png\n')

    # ── 4. SimCLR Pretraining Loop ──────────────────────────────────────
    print('Starting SimCLR Pretraining...')
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for (x_i, x_j), _ in ssl_loader:
            # Stack positive pairs: shape is (2 * batch_size, 3, 32, 32)
            x_batch = torch.cat([x_i, x_j], dim=0).to(device)
            
            optimizer.zero_grad()
            z_batch = model(x_batch)
            loss = nt_xent_loss(z_batch, temperature=TEMPERATURE)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(ssl_loader)
        loss_history.append(epoch_loss)
        print(f'Epoch [{epoch:02d}/{EPOCHS}] - Loss: {epoch_loss:.4f}')

    # Save Pretraining Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, label='Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('NT-Xent Loss')
    plt.title('SimCLR Unsupervised Pretraining Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/simclr_pretraining_loss.png', dpi=150)
    plt.close()
    print('\nPretraining loss curve saved → graphs/simclr_pretraining_loss.png')

    # ── 5. Save Encoder Weight Checkpoint ───────────────────────────────
    torch.save(model.encoder.state_dict(), 'models/simclr_encoder.pt')
    print('Encoder weights saved → models/simclr_encoder.pt')

    # ── 6. Compute Trained Similarities ─────────────────────────────────
    print('\nComputing average same/different image similarities (After Training)...')
    post_same, post_diff = compute_avg_similarities(model, sim_loader, device)
    print(f'After Training:')
    print(f'  Average Same-Image Similarity     : {post_same:.4f}')
    print(f'  Average Different-Image Similarity: {post_diff:.4f}\n')
    save_similarity_metrics(pre_same, pre_diff, post_same, post_diff, batch_size=BATCH_SIZE)
    print('After-training similarities saved → results/metrics.json\n')

    # Heatmap after training
    visualize_4_samples(
        model, ssl_dataset, device, 
        'results/similarity_matrix_after_training.png',
        "Cosine Similarity Matrix After Training\n(Pretrained ResNet-18 + MLP head)"
    )
    print('After-training heatmap saved → results/similarity_matrix_after_training.png')


