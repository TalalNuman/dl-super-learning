"""
MSDS25011 - Assignment 5, Task 1: Supervised Baseline
Loads CIFAR-10 with predefined train/val/test splits and trains a
ResNet-18 (modified for CIFAR-10) using only the 10 % labeled subset.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


# ── Model ───────────────────────────────────────────────────────────────────
def build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for 32×32 CIFAR-10 images.

    Modifications vs. ImageNet default:
      - conv1: 3×3 kernel, stride 1, padding 1 (instead of 7×7 / stride 2)
      - maxpool removed (replaced with nn.Identity)
      - fc head outputs `num_classes` (10)
    """
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


# ── Training helpers ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch; return average loss."""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model; return (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        running_loss += criterion(logits, labels).item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(loader.dataset)
    return running_loss / n, correct / n, all_preds, all_labels


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── Hyperparameters ─────────────────────────────────────────────────
    BATCH_SIZE = 64
    LR         = 3e-4
    EPOCHS     = 30
    SPLIT_DIR  = 'splits'

    # Output directories
    for d in ('graphs', 'results', 'models'):
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # ── 1. Load splits ──────────────────────────────────────────────────
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

    train_loader = DataLoader(splits['Train (10%)'], batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(splits['Validation'],  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(splits['Test'],        batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── 2. Model / loss / optimizer ─────────────────────────────────────
    model     = build_resnet18_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── 3. Training loop ────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_acc = 0.0

    print(f'\n{"Epoch":>5s} | {"Train Loss":>10s} | {"Val Loss":>10s} | {"Val Acc":>8s}')
    print('-' * 42)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_acc:7.2%}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f'\nBest validation accuracy: {best_val_acc:.2%}')

    # ── 4. Loss curve ───────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, EPOCHS + 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Supervised Baseline – Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/supervised_loss.png', dpi=150)
    plt.close()
    print('Loss curve saved → graphs/supervised_loss.png')

    # ── 5. Test evaluation ──────────────────────────────────────────────
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2%}')

    # ── 6. Confusion matrix ─────────────────────────────────────────────
    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=CIFAR10_CLASSES)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title('Supervised Baseline – Test Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/supervised_confusion_matrix.png', dpi=150)
    plt.close()
    print('Confusion matrix saved → results/supervised_confusion_matrix.png')

    # ── 7. Save model ───────────────────────────────────────────────────
    torch.save(model.state_dict(), 'models/supervised_model.pt')
    print('Model saved → models/supervised_model.pt')
