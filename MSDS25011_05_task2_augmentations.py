"""
Defines the SimCLR dual-view augmentation and visualises example pairs.
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ── CIFAR-10 normalization constants ────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ── SimCLR augmentation pipeline ───────────────────────────────────────────
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


# ── TwoViewTransform ───────────────────────────────────────────────────────
class TwoViewTransform:
    """Apply the same stochastic transform twice to produce two views."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


# ── Helpers ─────────────────────────────────────────────────────────────────
def denormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Reverse CIFAR-10 normalization for display (CHW tensor → HWC numpy)."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(3, 1, 1)
    img  = tensor * std + mean
    img  = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


# ── Main: generate augmentation visualisation ──────────────────────────────
if __name__ == '__main__':

    os.makedirs('results', exist_ok=True)

    # Load CIFAR-10 with the two-view transform
    two_view_tf = TwoViewTransform(simclr_transform)
    cifar_two_view = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=two_view_tf,
    )

    # Also load raw images (just ToTensor, no normalisation) for "Original"
    raw_transform = transforms.ToTensor()
    cifar_raw = datasets.CIFAR10(
        root='./data', train=True, download=False, transform=raw_transform,
    )

    # Pick 10 fixed indices spread across the dataset
    num_samples = 10
    indices = list(range(0, num_samples * 500, 500))  # 0, 500, 1000, ...

    # ── Build the grid ──────────────────────────────────────────────────
    fig, axes = plt.subplots(num_samples, 3, figsize=(7, 22))
    col_titles = ['Original Image', 'Augmented View 1', 'Augmented View 2']

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')

    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]

    for row, idx in enumerate(indices):
        # Original (unnormalised)
        raw_img, label = cifar_raw[idx]
        axes[row, 0].imshow(raw_img.permute(1, 2, 0).numpy())
        axes[row, 0].set_ylabel(CIFAR10_CLASSES[label], fontsize=9,
                                rotation=0, labelpad=50, va='center')

        # Two augmented views (denormalise for display)
        (v1, v2), _ = cifar_two_view[idx]
        axes[row, 1].imshow(denormalize(v1))
        axes[row, 2].imshow(denormalize(v2))

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('SimCLR Augmentation Examples', fontsize=14,
                 fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.05, 0, 1, 0.99])
    plt.savefig('results/augmentation_examples.png', dpi=150)
    plt.close()

    print('Augmentation visualisation saved → results/augmentation_examples.png')
