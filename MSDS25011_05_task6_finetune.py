"""
MSDS25011 - Assignment 5, Task 6: Fine-tuning SimCLR Encoder
Load a SimCLR-pretrained encoder and train the full model end-to-end
on the 10% labeled CIFAR-10 split for classification.
"""

import csv
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


def read_split_indices(split_file: str) -> list[int]:
    with open(split_file, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def build_encoder() -> nn.Module:
    """ResNet-18 encoder for CIFAR-10 (512-d output)."""
    encoder = models.resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.Identity()
    encoder.fc = nn.Identity()
    return encoder


class FinetuneModel(nn.Module):
    """SimCLR encoder + trainable linear classification head."""

    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
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
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        running_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n


@torch.no_grad()
def collect_test_predictions(
    model,
    loader: DataLoader,
    image_indices: list[int],
    device: torch.device,
) -> tuple[float, list[dict]]:
    model.eval()
    rows: list[dict] = []
    correct = 0
    total = 0
    offset = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()

        for i in range(len(true_labels)):
            row = {
                'image_index': image_indices[offset + i],
                'true_label': int(true_labels[i]),
                'predicted_label': int(preds[i]),
            }
            for c in range(10):
                row[f'prob_class_{c}'] = float(probs[i, c])
            rows.append(row)

        correct += (preds == true_labels).sum()
        total += len(true_labels)
        offset += len(true_labels)

    return correct / total, rows


def save_test_predictions_csv(rows: list[dict], out_path: str) -> None:
    fieldnames = (
        ['image_index', 'true_label', 'predicted_label']
        + [f'prob_class_{c}' for c in range(10)]
    )
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    test_indices = read_split_indices(os.path.join(SPLIT_DIR, 'test.txt'))

    model = FinetuneModel().to(device)
    encoder_state = torch.load(
        'models/simclr_encoder.pt', map_location=device, weights_only=True,
    )
    model.encoder.load_state_dict(encoder_state)
    print('Loaded SimCLR encoder weights from models/simclr_encoder.pt\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    val_accs: list[float] = []
    print(f'{"Epoch":>5s} | {"Train Loss":>10s} | {"Val Loss":>10s} | {"Val Acc":>8s}')
    print('-' * 42)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_accs.append(val_acc)
        print(f'{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_acc:7.2%}')

    test_acc, prediction_rows = collect_test_predictions(
        model, test_loader, test_indices, device,
    )
    print(f'\nTest Accuracy: {test_acc:.2%}')

    torch.save(model.state_dict(), 'models/finetuned_model.pt')
    print('Model saved → models/finetuned_model.pt')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('SimCLR Fine-tuning — Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/finetuning_accuracy.png', dpi=150)
    plt.close()
    print('Accuracy curve saved → graphs/finetuning_accuracy.png')

    save_test_predictions_csv(prediction_rows, 'results/test_predictions.csv')
    print('Test predictions saved → results/test_predictions.csv')
