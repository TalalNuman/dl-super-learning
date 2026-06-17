"""
MSDS25011 - Assignment 5, Task 5: Linear Probe Evaluation
Freeze a ResNet-18 encoder and train only a linear classifier on 10% labels.
Compares a random encoder (Experiment A) vs SimCLR-pretrained encoder (Experiment B).
"""

import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── CIFAR-10 normalization ──────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


def load_split(split_file: str, cifar_root: str = './data') -> Subset:
    """Load a CIFAR-10 subset from a fixed index file."""
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
    """ResNet-18 encoder for CIFAR-10 (512-d output, no classification head)."""
    encoder = models.resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.Identity()
    encoder.fc = nn.Identity()
    return encoder


class LinearProbeModel(nn.Module):
    """Frozen encoder + trainable linear classifier."""

    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.classifier = nn.Linear(512, 10)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.classifier.train()
    model.encoder.eval()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.encoder.eval()
    model.classifier.eval()
    running_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        running_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n


def run_linear_probe(
    experiment_name: str,
    encoder_weights_path: str | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> tuple[list[float], float]:
    """Train a linear probe and return per-epoch val accuracies and test accuracy."""
    set_seed()
    model = LinearProbeModel().to(device)

    if encoder_weights_path is not None:
        state_dict = torch.load(encoder_weights_path, map_location=device, weights_only=True)
        model.encoder.load_state_dict(state_dict)
        print(f'Loaded encoder weights from {encoder_weights_path}')
    else:
        print('Using randomly initialized encoder')

    model.freeze_encoder()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_accs: list[float] = []
    print(f'\n{experiment_name}')
    print(f'{"Epoch":>5s} | {"Train Loss":>10s} | {"Val Loss":>10s} | {"Val Acc":>8s}')
    print('-' * 42)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_accs.append(val_acc)
        print(f'{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_acc:7.2%}')

    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Final test accuracy: {test_acc:.2%}\n')
    return model, val_accs, test_acc


if __name__ == '__main__':
    BATCH_SIZE = 64
    LR = 3e-4
    EPOCHS = 20
    SPLIT_DIR = 'splits'

    for d in ('graphs', 'results', 'models'):
        os.makedirs(d, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}\n')

    train_loader = DataLoader(
        load_split(os.path.join(SPLIT_DIR, 'train_labeled_10percent.txt')),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        load_split(os.path.join(SPLIT_DIR, 'val.txt')),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        load_split(os.path.join(SPLIT_DIR, 'test.txt')),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    random_model, random_val_accs, random_test_acc = run_linear_probe(
        experiment_name='Experiment A — Random Encoder Linear Probe',
        encoder_weights_path=None,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
    )

    simclr_model, simclr_val_accs, simclr_test_acc = run_linear_probe(
        experiment_name='Experiment B — SimCLR Encoder Linear Probe',
        encoder_weights_path='models/simclr_encoder.pt',
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
    )

    torch.save(simclr_model.state_dict(), 'models/linear_probe.pt')
    print('SimCLR linear probe saved → models/linear_probe.pt')

    plt.figure(figsize=(8, 5))
    epochs_range = range(1, EPOCHS + 1)
    plt.plot(epochs_range, random_val_accs, label='Random Encoder')
    plt.plot(epochs_range, simclr_val_accs, label='SimCLR Encoder')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Linear Probe — Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/linear_probe_accuracy.png', dpi=150)
    plt.close()
    print('Validation accuracy plot saved → graphs/linear_probe_accuracy.png')

    print('=' * 42)
    print('Final Test Accuracies')
    print('=' * 42)
    print(f'Random Encoder Linear Probe : {random_test_acc:.2%}')
    print(f'SimCLR Encoder Linear Probe : {simclr_test_acc:.2%}')
