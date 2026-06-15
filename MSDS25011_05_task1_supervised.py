"""
MSDS25011 - Assignment 5, Task 1: Supervised Baseline
Loads CIFAR-10 with predefined train/val/test splits.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

# ── Reproducibility: seed everything ────────────────────────────────────────
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ── CIFAR-10 normalization ──────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


# ── Core helper ─────────────────────────────────────────────────────────────
def load_split(split_file: str, cifar_root: str = './data') -> Subset:
    """
    Load a CIFAR-10 subset defined by an index file.

    Parameters
    ----------
    split_file : str
        Path to a .txt file containing one integer index per line.
    cifar_root : str
        Root directory for CIFAR-10 data (downloaded here if absent).

    Returns
    -------
    torch.utils.data.Subset
        The subset of CIFAR-10 corresponding to the given indices.
    """
    # Determine if we need the train or test portion of CIFAR-10
    # test.txt indices refer to the CIFAR-10 test set; everything else
    # refers to the training set.
    is_train = 'test' not in os.path.basename(split_file)

    full_dataset = datasets.CIFAR10(
        root=cifar_root,
        train=is_train,
        download=True,
        transform=transform_cifar,
    )

    # Read indices
    with open(split_file, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip()]

    return Subset(full_dataset, indices)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    SPLIT_DIR = 'splits'

    split_files = {
        'Train (10%)': os.path.join(SPLIT_DIR, 'train_labeled_10percent.txt'),
        'Validation' : os.path.join(SPLIT_DIR, 'val.txt'),
        'Test'       : os.path.join(SPLIT_DIR, 'test.txt'),
    }

    splits = {}
    for name, path in split_files.items():
        subset = load_split(path)
        splits[name] = subset
        print(f'{name:>12s}  →  {len(subset):,} samples')

    print('\nAll splits loaded successfully.')
