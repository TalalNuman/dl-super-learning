"""
MSDS25011 - Assignment 5, Task 5: SimCLR Pretraining
Unsupervised contrastive pretraining on the SSL split (no labels used).
"""

import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from MSDS25011_05_task2_augmentations import TwoViewTransform, simclr_transform
from MSDS25011_05_task4_simclr import (
    SimCLRModel,
    load_split,
    nt_xent_loss,
    visualize_4_samples,
)

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    BATCH_SIZE = 64
    LR = 3e-4
    EPOCHS = 50
    TEMPERATURE = 0.5
    SSL_SPLIT = 'splits/train_ssl_unlabeled.txt'

    for d in ('graphs', 'results', 'models'):
        os.makedirs(d, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}\n')

    print('Loading unlabeled SSL subset...')
    two_view_tf = TwoViewTransform(simclr_transform)
    ssl_dataset = load_split(SSL_SPLIT, transform=two_view_tf)
    print(f'SSL pretraining samples: {len(ssl_dataset):,}\n')

    ssl_loader = torch.utils.data.DataLoader(
        ssl_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True,
    )

    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('Starting SimCLR Pretraining...')
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for (x_i, x_j), _ in ssl_loader:
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

    torch.save(model.encoder.state_dict(), 'models/simclr_encoder.pt')
    print('Encoder weights saved → models/simclr_encoder.pt')

    visualize_4_samples(
        model, ssl_dataset, device,
        'results/similarity_matrix_after_training.png',
        'Cosine Similarity Matrix After Training\n(Pretrained ResNet-18 + MLP head)',
    )
    print('After-training heatmap saved → results/similarity_matrix_after_training.png')
