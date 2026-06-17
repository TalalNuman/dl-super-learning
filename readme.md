# MSDS25011 – Assignment 5: SimCLR & Supervised Baseline

**Student:** Talal · **Roll number:** MSDS25011 · **Seed:** 2026

## Project Structure

```
AS5/
├── MSDS25011_05_task1_supervised.py      # Task 1 – Supervised baseline (10% labels)
├── MSDS25011_05_task2_augmentations.py   # Task 2 – SimCLR augmentation pipeline
├── MSDS25011_05_task3_similarity.py      # Task 3 – Feature similarity before training
├── MSDS25011_05_task4_simclr.py          # Task 4 – SimCLR model, NT-Xent loss, before-training heatmap
├── MSDS25011_05_task5_pretraining.py     # Task 5 – SimCLR unsupervised pretraining
├── MSDS25011_05_task6_linear_probe.py    # Task 6 – Linear probe (random vs SimCLR encoder)
├── MSDS25011_05_task7_finetune.py        # Task 7 – Fine-tune SimCLR encoder end-to-end
├── MSDS25011_05_task8_visualization.py   # Task 8 – t-SNE feature visualization
├── splits/
│   ├── train_ssl_unlabeled.txt           # 45 000 unlabeled indices (SimCLR pretraining)
│   ├── train_labeled_10percent.txt       #  5 000 labeled training indices
│   ├── val.txt                           #  5 000 validation indices
│   └── test.txt                          # 10 000 test indices
├── data/                                 # CIFAR-10 (auto-downloaded)
├── graphs/
│   ├── supervised_loss.png
│   ├── simclr_pretraining_loss.png
│   ├── linear_probe_accuracy.png
│   └── finetuning_accuracy.png
├── results/
│   ├── augmentation_examples.png
│   ├── similarity_matrix_before_training.png
│   ├── similarity_matrix_after_training.png
│   ├── supervised_confusion_matrix.png
│   ├── random_encoder_pca_or_tsne.png
│   ├── simclr_encoder_pca_or_tsne.png
│   ├── finetuned_encoder_pca_or_tsne.png
│   ├── metrics.json
│   └── test_predictions.csv
├── models/
│   ├── supervised_model.pt
│   ├── simclr_encoder.pt
│   ├── linear_probe.pt
│   └── finetuned_model.pt
├── templates/
├── utils/
└── readme.md
```

## Recommended Run Order

Run from the `AS5/` directory. Each script uses **seed 2026** and **MPS if available, else CPU**.

```bash
python MSDS25011_05_task1_supervised.py
python MSDS25011_05_task2_augmentations.py
python MSDS25011_05_task3_similarity.py
python MSDS25011_05_task4_simclr.py
python MSDS25011_05_task5_pretraining.py      # ~50 epochs — slowest step
python MSDS25011_05_task6_linear_probe.py
python MSDS25011_05_task7_finetune.py
python MSDS25011_05_task8_visualization.py
```

---

## Global Settings

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 |
| Random seed | 2026 |
| Normalization mean | (0.4914, 0.4822, 0.4465) |
| Normalization std | (0.2470, 0.2435, 0.2616) |
| Encoder | ResNet-18 modified for CIFAR-10 (3×3 conv1, no maxpool) |
| Device | MPS if available, else CPU |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Optimizer | Adam |

---

## Task 1 – Supervised Baseline (`MSDS25011_05_task1_supervised.py`)

Train a ResNet-18 classifier from scratch on the **10% labeled** split.

| Parameter | Value |
|-----------|-------|
| Splits | `train_labeled_10percent.txt`, `val.txt`, `test.txt` |
| Epochs | 30 |
| Loss | CrossEntropyLoss |

### Outputs

| File | Description |
|------|-------------|
| `graphs/supervised_loss.png` | Train & validation loss over 30 epochs |
| `results/supervised_confusion_matrix.png` | 10×10 test-set confusion matrix |
| `models/supervised_model.pt` | Trained ResNet-18 weights |

### Run

```bash
python MSDS25011_05_task1_supervised.py
```

### Checklist

- [ ] Split sizes print as 5 000 / 5 000 / 10 000
- [ ] 30 epochs of loss & val accuracy printed
- [ ] Test accuracy ~55–65% (10% labels)
- [ ] All three output files exist

---

## Task 2 – SimCLR Augmentations (`MSDS25011_05_task2_augmentations.py`)

Visualize the SimCLR augmentation pipeline and `TwoViewTransform` wrapper.

| Transform | Parameters |
|-----------|------------|
| `RandomResizedCrop` | size=32, scale=(0.2, 1.0) |
| `RandomHorizontalFlip` | p=0.5 |
| `ColorJitter` | brightness/contrast/saturation=0.4, hue=0.1 |
| `RandomGrayscale` | p=0.2 |
| `ToTensor` + `Normalize` | CIFAR-10 stats |

`TwoViewTransform` applies the pipeline **twice independently**, returning `(view1, view2)`.

### Output

| File | Description |
|------|-------------|
| `results/augmentation_examples.png` | 10 rows × 3 cols: Original \| View 1 \| View 2 |

### Run

```bash
python MSDS25011_05_task2_augmentations.py
```

### Checklist

- [ ] Grid is 10 × 3 with visibly different augmented views per row
- [ ] Original column shows un-normalised CIFAR-10 images

---

## Task 3 – Feature Similarity Before Training (`MSDS25011_05_task3_similarity.py`)

Pass validation images through a **random untrained** `SimCLRModel` encoder and measure average cosine similarity between:

- **Same-image pairs** — view1[i] vs view2[i] (diagonal)
- **Different-image pairs** — view1[i] vs view2[j], i ≠ j (off-diagonal)

Uses `splits/val.txt` with `TwoViewTransform`. Labels are **not** used.

### Run

```bash
python MSDS25011_05_task3_similarity.py
```

### Expected console output

```
Computing average same/different image similarities (Before Training)...
  Average Same-Image Similarity     : ~0.99
  Average Different-Image Similarity: ~0.99
```

Before SimCLR training, same- and different-image similarities should both be high (random encoder has not yet learned to separate views).

### Checklist

- [ ] Both similarity values printed
- [ ] Same-image and different-image similarities are similar before training

---

## Task 4 – SimCLR Architecture & NT-Xent Loss (`MSDS25011_05_task4_simclr.py`)

Defines the SimCLR model and contrastive loss (implemented from scratch, no external NT-Xent library).

### Model – SimCLRModel

| Component | Detail |
|-----------|--------|
| **Encoder** | ResNet-18 (3×3 conv1, no maxpool, fc removed) → 512-d features |
| **Projector** | Linear(512→256) → ReLU → Linear(256→128) → 128-d projection |

### Functions

| Function | Description |
|----------|-------------|
| `compute_similarity_matrix(z)` | L2-normalize rows, return (2N, 2N) cosine similarity matrix |
| `nt_xent_loss(z, temperature=0.5)` | Vectorized NT-Xent; positive pair for sample i is i+N |

### Output

| File | Description |
|------|-------------|
| `results/similarity_matrix_before_training.png` | 8×8 heatmap for 4 images × 2 views (untrained model) |

### Run

```bash
python MSDS25011_05_task4_simclr.py
```

### Checklist

- [ ] Heatmap is 8×8 with labels `Img1_V1` … `Img4_V2`
- [ ] Diagonal ≈ 1.0; off-diagonal values low under random weights

---

## Task 5 – SimCLR Pretraining (`MSDS25011_05_task5_pretraining.py`)

Unsupervised contrastive pretraining on `train_ssl_unlabeled.txt` (45 000 images). **Labels are not used.**

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Temperature τ | 0.5 |
| Loss | NT-Xent (from Task 4) |

Imports `SimCLRModel`, `nt_xent_loss`, and `TwoViewTransform` from Tasks 4 and 2.

### Outputs

| File | Description |
|------|-------------|
| `graphs/simclr_pretraining_loss.png` | NT-Xent loss over 50 epochs |
| `models/simclr_encoder.pt` | Pretrained encoder weights |
| `results/similarity_matrix_after_training.png` | 8×8 heatmap after pretraining |

### Run

```bash
python MSDS25011_05_task5_pretraining.py
```

### Checklist

- [ ] Loss decreases over 50 epochs
- [ ] `models/simclr_encoder.pt` saved
- [ ] After-training heatmap shows higher same-image / lower different-image similarity in projection space

---

## Task 6 – Linear Probe Evaluation (`MSDS25011_05_task6_linear_probe.py`)

Freeze the encoder and train **only** `Linear(512, 10)` on the 10% labeled split.

| Experiment | Encoder | Trainable |
|------------|---------|-----------|
| **A — Random** | Untrained, frozen | Classifier only |
| **B — SimCLR** | `models/simclr_encoder.pt`, frozen | Classifier only |

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Splits | `train_labeled_10percent.txt`, `val.txt`, `test.txt` |

### Outputs

| File | Description |
|------|-------------|
| `graphs/linear_probe_accuracy.png` | Val accuracy curves for both experiments |
| `models/linear_probe.pt` | SimCLR linear probe state dict |

### Run

```bash
python MSDS25011_05_task6_linear_probe.py
```

### Checklist

- [ ] Val accuracy printed per epoch for both experiments
- [ ] SimCLR encoder probe beats random encoder probe on test set
- [ ] Plot and checkpoint saved

---

## Task 7 – Fine-tuning (`MSDS25011_05_task7_finetune.py`)

Load SimCLR encoder weights and train the **full model** (encoder + classifier) end-to-end on 10% labels.

| Parameter | Value |
|-----------|-------|
| Init | `models/simclr_encoder.pt` |
| Epochs | 20 |
| Loss | CrossEntropyLoss |

### Outputs

| File | Description |
|------|-------------|
| `graphs/finetuning_accuracy.png` | Validation accuracy over 20 epochs |
| `models/finetuned_model.pt` | Full fine-tuned model state dict |
| `results/test_predictions.csv` | Test predictions with class probabilities |

`test_predictions.csv` columns: `image_index, true_label, predicted_label, prob_class_0 … prob_class_9`

### Run

```bash
python MSDS25011_05_task7_finetune.py
```

### Checklist

- [ ] Train / val loss and val accuracy printed each epoch
- [ ] Test accuracy printed
- [ ] All three output files exist

---

## Task 8 – t-SNE Feature Visualization (`MSDS25011_05_task8_visualization.py`)

Extract **512-d encoder features** from **1 000 validation images** (fixed subset, seed=2026) and reduce to 2D with t-SNE. Labels are used **only for coloring**.

| Encoder | Weights |
|---------|---------|
| Random untrained | None |
| SimCLR pretrained | `models/simclr_encoder.pt` |
| Fine-tuned | `models/finetuned_model.pt` (encoder only) |

### Outputs

| File | Description |
|------|-------------|
| `results/random_encoder_pca_or_tsne.png` | t-SNE of random encoder features |
| `results/simclr_encoder_pca_or_tsne.png` | t-SNE of SimCLR encoder features |
| `results/finetuned_encoder_pca_or_tsne.png` | t-SNE of fine-tuned encoder features |

### Run

```bash
python MSDS25011_05_task8_visualization.py
```

Requires Task 7 to be complete for the fine-tuned plot.

### Checklist

- [ ] Three plots saved with distinct colors and class-name legend
- [ ] SimCLR and fine-tuned plots show clearer class grouping than random encoder

---

## Final Results

### Test accuracy comparison

| Model | Labels in pretraining? | Encoder frozen? | Test accuracy |
|-------|------------------------|-----------------|---------------|
| Supervised ResNet-18 (10% labels) | Yes | No | 53.72% |
| Random encoder + linear probe | No | Yes | — |
| SimCLR encoder + linear probe | No | Yes | 74.84% |
| SimCLR encoder + fine-tuning | No / Yes (finetune) | No | 78.70% |

### Feature similarity (encoder features, validation set)

| Pair type | Before SimCLR | After SimCLR |
|-----------|---------------|--------------|
| Same image, two views | 0.989 | 0.917 |
| Different images | 0.985 | 0.436 |

Values from `results/metrics.json`. After pretraining, same-image similarity stays relatively high while different-image similarity drops — the encoder learns to align views of the same image and separate different images.

---

## metrics.json

`results/metrics.json` records hyperparameters and final numbers in the assignment-required format:

```json
{
  "student_name": "Talal",
  "roll_number": "MSDS25011",
  "seed": 2026,
  "batch_size": 64,
  "simclr_epochs": 50,
  "linear_probe_epochs": 20,
  "finetuning_epochs": 20,
  "learning_rate": 0.0003,
  "temperature": 0.5,
  "supervised_10percent_test_acc": 0.5372,
  "random_linear_probe_test_acc": 0.0,
  "simclr_linear_probe_test_acc": 0.7484,
  "simclr_finetune_test_acc": 0.787,
  "same_view_similarity_before": 0.988712,
  "different_image_similarity_before": 0.985377,
  "same_view_similarity_after": 0.916729,
  "different_image_similarity_after": 0.435537
}
```

Update `random_linear_probe_test_acc` after running Task 6 Experiment A.

---

## Dependencies

- Python 3.10+
- PyTorch & torchvision
- numpy, matplotlib, scikit-learn
