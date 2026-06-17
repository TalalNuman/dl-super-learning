# Assignment 5 — SimCLR

CIFAR-10 self-supervised learning with SimCLR and a supervised baseline. All scripts use **seed 2026** and **MPS if available, else CPU**.

## Scripts

| File | What it does |
|------|--------------|
| `MSDS25011_05_task1_supervised.py` | Train ResNet-18 on 10% labeled data (supervised baseline) |
| `MSDS25011_05_task2_augmentations.py` | SimCLR augmentation pipeline + two-view examples |
| `MSDS25011_05_task3_similarity.py` | Avg cosine similarity (same vs different images) before training |
| `MSDS25011_05_task4_simclr.py` | SimCLR model, NT-Xent loss, before-training similarity heatmap |
| `MSDS25011_05_task5_pretraining.py` | SimCLR pretraining on unlabeled data (50 epochs) |
| `MSDS25011_05_task6_linear_probe.py` | Linear probe: random vs SimCLR frozen encoder |
| `MSDS25011_05_task7_finetune.py` | Fine-tune SimCLR encoder end-to-end on 10% labels |
| `MSDS25011_05_task8_visualization.py` | t-SNE plots for random, SimCLR, and fine-tuned encoders |

## How to run

Run from this directory, in order:

```bash
python MSDS25011_05_task1_supervised.py
python MSDS25011_05_task2_augmentations.py
python MSDS25011_05_task3_similarity.py
python MSDS25011_05_task4_simclr.py
python MSDS25011_05_task5_pretraining.py
python MSDS25011_05_task6_linear_probe.py
python MSDS25011_05_task7_finetune.py
python MSDS25011_05_task8_visualization.py
```

Task 5 is the slowest (50 epochs). Task 8 needs Task 7 finished first.

## Main outputs

**Graphs** — `graphs/supervised_loss.png`, `simclr_pretraining_loss.png`, `linear_probe_accuracy.png`, `finetuning_accuracy.png`

**Results** — `results/augmentation_examples.png`, `similarity_matrix_before_training.png`, `similarity_matrix_after_training.png`, `supervised_confusion_matrix.png`, `random_encoder_pca_or_tsne.png`, `simclr_encoder_pca_or_tsne.png`, `finetuned_encoder_pca_or_tsne.png`, `test_predictions.csv`, `metrics.json`

**Models** — `models/supervised_model.pt`, `simclr_encoder.pt`, `linear_probe.pt`, `finetuned_model.pt`

## Notes

- Data splits live in `splits/` (fixed indices — do not use `random_split`)
- CIFAR-10 downloads automatically to `data/` on first run
- Shared helpers are in `utils/`
