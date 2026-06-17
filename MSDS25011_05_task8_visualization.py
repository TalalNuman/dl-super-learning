"""
MSDS25011 - Assignment 5, Task 8: Feature Visualization (t-SNE)
Extract 512-d encoder features from 1000 validation images and visualize
with t-SNE for random, SimCLR-pretrained, and fine-tuned encoders.
"""

import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from MSDS25011_05_task7_finetune import FinetuneModel

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 2026
NUM_VIS_SAMPLES = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


def load_split(split_file: str, cifar_root: str = './data') -> Subset:
    is_train = 'test' not in os.path.basename(split_file)
    full_dataset = datasets.CIFAR10(
        root=cifar_root,
        train=is_train,
        download=True,
        transform=transform_cifar,
    )
    with open(split_file, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    return Subset(full_dataset, indices)


def build_encoder() -> nn.Module:
    encoder = models.resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.Identity()
    encoder.fc = nn.Identity()
    return encoder


def select_fixed_val_subset(val_dataset: Subset, n: int = NUM_VIS_SAMPLES, seed: int = SEED) -> Subset:
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(val_dataset), size=n, replace=False))
    return Subset(val_dataset, chosen.tolist())


@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader, device: torch.device):
    encoder.eval()
    features, labels = [], []
    for images, lbls in loader:
        images = images.to(device)
        h = encoder(images)
        features.append(h.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def save_tsne_plot(
    features: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
    seed: int = SEED,
) -> None:
    coords = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        perplexity=30,
        random_state=seed,
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=10, alpha=0.75, label=class_name,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(title='Class', loc='best', markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    BATCH_SIZE = 64

    os.makedirs('results', exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}\n')

    val_dataset = load_split('splits/val.txt')
    vis_subset = select_fixed_val_subset(val_dataset, n=NUM_VIS_SAMPLES, seed=SEED)
    vis_loader = DataLoader(vis_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'Visualization subset: {len(vis_subset):,} validation images (seed={SEED})\n')

    plots = [
        (
            'Random Untrained Encoder',
            build_encoder(),
            None,
            'results/random_encoder_pca_or_tsne.png',
        ),
        (
            'SimCLR Pretrained Encoder',
            build_encoder(),
            'models/simclr_encoder.pt',
            'results/simclr_encoder_pca_or_tsne.png',
        ),
        (
            'Fine-tuned Encoder',
            build_encoder(),
            'models/finetuned_model.pt',
            'results/finetuned_encoder_pca_or_tsne.png',
        ),
    ]

    for title, encoder, weights_path, out_path in plots:
        if weights_path == 'models/finetuned_model.pt':
            finetune_path = 'models/finetuned_model.pt'
            if not os.path.exists(finetune_path):
                print(f'Skipping {title}: {finetune_path} not found (run Task 7).')
                continue
            finetune_model = FinetuneModel().to(device)
            finetune_model.load_state_dict(torch.load(
                finetune_path, map_location=device, weights_only=True,
            ))
            encoder = finetune_model.encoder
        else:
            encoder = encoder.to(device)
            if weights_path is not None:
                encoder.load_state_dict(torch.load(
                    weights_path, map_location=device, weights_only=True,
                ))

        print(f'Extracting features — {title}...')
        features, labels = extract_features(encoder, vis_loader, device)
        print(f'  Feature shape: {features.shape}')

        print(f'  Running t-SNE → {out_path}')
        save_tsne_plot(features, labels, out_path, title=f'{title} — t-SNE')
        print(f'  Saved {out_path}\n')
