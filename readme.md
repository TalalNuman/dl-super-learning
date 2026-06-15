# MSDS25011 – Assignment 5: SimCLR & Supervised Baseline

## Project Structure

```
AS5/
├── MSDS25011_05_task1_supervised.py   # Task 1 – Supervised baseline (load + train + eval)
├── MSDS25011_05_task2_augmentations.py # Task 2 – SimCLR augmentation pipeline & visualisation
├── MSDS25011_05_task4_simclr.py       # Task 4 – SimCLR network, similarity matrix & NT-Xent loss
├── splits/
│   ├── train_labeled_10percent.txt    # 5 000 labeled training indices
│   ├── val.txt                        # 5 000 validation indices
│   └── test.txt                       # 10 000 test indices
├── data/                              # CIFAR-10 data (auto-downloaded)
├── graphs/
│   └── supervised_loss.png            # Train & val loss curves
├── results/
│   ├── supervised_confusion_matrix.png # Test-set confusion matrix
│   ├── augmentation_examples.png      # SimCLR augmentation grid (10×3)
│   └── similarity_matrix_before_training.png # Cosine similarity heatmap of 4 samples (8 views)
├── models/
│   └── supervised_model.pt            # Trained ResNet-18 weights
├── templates/
├── utils/
└── readme.md                          # ← you are here
```

---

## Steps Completed

### Step 1 – Supervised Data Loading (`MSDS25011_05_task1_supervised.py`)

| What | Detail |
|------|--------|
| **Random seed** | `2026` set for `random`, `numpy`, `torch` (CPU + CUDA), `cudnn` deterministic mode enabled |
| **Normalization** | CIFAR-10 channel-wise: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) |
| **`load_split()`** | Reads a `.txt` index file → returns a `torch.utils.data.Subset` (no `random_split`) |
| **Splits loaded** | `train_labeled_10percent.txt` (5 000), `val.txt` (5 000), `test.txt` (10 000) |

---

### Step 2 – Supervised Training & Evaluation (`MSDS25011_05_task1_supervised.py`)

#### Model – ResNet-18 (CIFAR-10 variant)

| Layer | Modification |
|-------|-------------|
| `conv1` | `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)` |
| `maxpool` | Replaced with `nn.Identity()` |
| `fc` | `nn.Linear(512, 10)` |

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Epochs | 30 |
| Batch size | 64 |
| Loss | CrossEntropyLoss |
| Seed | 2026 |

#### Outputs

| File | Description |
|------|-------------|
| `graphs/supervised_loss.png` | Train & validation loss plotted over 30 epochs |
| `results/supervised_confusion_matrix.png` | 10×10 confusion matrix on test set (sklearn) |
| `models/supervised_model.pt` | Saved `state_dict` of trained ResNet-18 |

#### How to Verify

```bash
# From the AS5/ directory — runs the full pipeline (data → train → eval → save)
python MSDS25011_05_task1_supervised.py
```

**Expected console output:**

```
Using device: cuda          # or cpu

 Train (10%)  →  5,000 samples
  Validation  →  5,000 samples
        Test  →  10,000 samples

Epoch | Train Loss |   Val Loss |  Val Acc
------------------------------------------
    1 |     2.xxxx |     1.xxxx |  xx.xx%
  ...
   30 |     0.xxxx |     1.xxxx |  xx.xx%

Best validation accuracy: xx.xx%
Loss curve saved → graphs/supervised_loss.png

Test Loss: x.xxxx
Test Accuracy: xx.xx%
Confusion matrix saved → results/supervised_confusion_matrix.png
Model saved → models/supervised_model.pt
```

**Checklist after running:**

- [ ] Three split sizes print correctly (5 000 / 5 000 / 10 000)
- [ ] 30 epochs of loss & accuracy are printed
- [ ] `graphs/supervised_loss.png` exists and shows decreasing train loss
- [ ] `results/supervised_confusion_matrix.png` exists with 10×10 grid
- [ ] `models/supervised_model.pt` exists
- [ ] Test accuracy is reported (expect ~55-65 % with only 10 % labels, CPU/GPU may vary slightly)

---

### Step 3 – SimCLR Augmentations (`MSDS25011_05_task2_augmentations.py`)

#### Augmentation Pipeline

| Transform | Parameters |
|-----------|------------|
| `RandomResizedCrop` | size=32, scale=(0.2, 1.0) |
| `RandomHorizontalFlip` | p=0.5 |
| `ColorJitter` | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 |
| `RandomGrayscale` | p=0.2 |
| `ToTensor` | — |
| `Normalize` | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) |

#### TwoViewTransform

Applies the above pipeline **twice independently** to the same input image, returning `(view1, view2)` — the positive pair used by SimCLR contrastive learning.

#### Output

| File | Description |
|------|-------------|
| `results/augmentation_examples.png` | 10 rows × 3 cols grid: Original \| View 1 \| View 2 |

#### How to Verify

```bash
python MSDS25011_05_task2_augmentations.py
```

**Expected output:**

```
Augmentation visualisation saved → results/augmentation_examples.png
```

**Checklist:**

- [ ] `results/augmentation_examples.png` exists
- [ ] Grid shows 10 rows × 3 columns (Original | View 1 | View 2)
- [ ] Each row's two views look visibly different from each other (crops, flips, colour shifts)
- [ ] Original images look like normal CIFAR-10 (not normalised)

---

### Step 4 – SimCLR Architecture & Contrastive Loss (`MSDS25011_05_task4_simclr.py`)

#### Model – SimCLRModel

- **Encoder**: ResNet-18 modified for CIFAR-10:
  - `conv1`: `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`
  - `maxpool`: Replaced with `nn.Identity()`
  - Final fc layer removed (output is 512-dimensional feature vector $h$).
- **Projection Head**: Two-layer MLP (used only during pretraining):
  - `nn.Linear(512, 256)`
  - `nn.ReLU()`
  - `nn.Linear(256, 128)`
  - Output is 128-dimensional projection vector $z$.

#### Contrastive Loss Implementation
- **`compute_similarity_matrix(z)`**: Computes the cosine similarity matrix of shape $(2N, 2N)$ for a batch of $2N$ views.
- **`nt_xent_loss(z, temperature=0.5)`**: A completely vectorized implementation from scratch (without loops or external libraries) of Normalized Temperature-scaled Cross Entropy Loss. Masks out self-similarity (diagonal) and maps target indices for cross-entropy optimization.

#### Pre-training Visualization
- Generates a similarity heatmap for a small batch of $N=4$ images ($2N=8$ views) under an untrained, randomized network.
- Output file: `results/similarity_matrix_before_training.png`

#### How to Verify

```bash
python MSDS25011_05_task4_simclr.py
```

**Expected output:**
- Prints detailed module structure of `model.encoder` and `model.projector`.
- Verifies parameter counts:
  - Encoder: 11,168,832 params
  - Projector: 164,224 params
  - Total: 11,333,056 params
- Performs a dummy forward pass validation with shapes:
  - `forward()` output shape: `(2, 128)`
  - `get_features()` output shape: `(2, 512)`
- Loads CIFAR-10, applies the TwoViewTransform, runs the untrained model forward pass, prints the batch loss, and saves the similarity heatmap:
  - `NT-Xent Loss on this batch: 1.8725` (values close to 1.8-2.1 are expected under random weights).
  - Heatmap saved to `results/similarity_matrix_before_training.png`.

**Checklist:**
- [ ] `results/similarity_matrix_before_training.png` exists.
- [ ] Heatmap contains an $8\times8$ grid with axis labels `Img1_V1` ... `Img4_V2`.
- [ ] Diagonal is 1.0 (self-similarity) and off-diagonal values are low (random distribution) before training.

