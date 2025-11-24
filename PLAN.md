# Machine Unlearning with SISA — Project Plan

## Objectives
- Implement a SISA (Sharded, Isolated, Sliced, Aggregated) training pipeline in PyTorch for CIFAR-10 using a ≤1M parameter CNN.
- Support selective retraining when handling deletion requests while minimizing full retraining cost.
- Quantify trade-offs between accuracy, wall-clock time, and privacy risk via a lightweight membership inference attack (MIA).
- Produce reproducible experiments, logs, plots, and a concise report that contextualizes findings for adapter/LoRA fine-tunes in FM workflows.

## Dataset & Model Choices
- **Dataset:** CIFAR-10 via `torchvision.datasets` for reproducible access and manageable size.
- **Model:** Compact CNN (~0.7M params) with Conv-BN-ReLU blocks, dropout, and global avg pooling; compatible with CPU/GPU.
- **Transforms:** Normalize inputs, apply light augmentation (random crop/flip) during training.

## SISA Configuration
- **Default:** `K=10` shards, `S=2` slices per shard. CLI flags allow overrides.
- **Sharding:** Deterministic split by shuffled indices. Each shard keeps its own sequential slices.
- **Isolation:** Each shard owns its model weights; slices train sequentially carrying previous slice weights.
- **Aggregation:** Aggregate shard heads by parameter-wise averaging to form final global model (optionally majority vote at eval time).

## Experiments
1. **Baseline (No Deletions):**
   - Train full SISA stack.
   - Log per-slice metrics (loss, accuracy, wall-clock, mem usage).
   - Evaluate aggregated model on test set (accuracy, F1).
2. **Deletion Scenarios:**
   - Simulate deletion requests for 1% and 5% of training records.
   - Identify affected slices (same shard + slice where data lived).
   - Retrain only impacted slices using stored checkpoints for preceding slice.
   - Record \u0394accuracy, time saved, fraction of slices retrained.
3. **Privacy / MIA:**
   - Implement confidence-score-based membership inference.
   - Compare AUC for: baseline model, naive delete (full retrain), and SISA-unlearned model after deletions.

## Metrics & Logging
- Accuracy, F1, per-slice loss curves.
- Time per slice/shard + total, GPU mem usage estimate.
- Fraction of slices retrained vs % deleted.
- CSV summaries stored under `results/metrics_*.csv`.
- Plots: `Fig-1` accuracy vs % deleted, `Fig-2` time saved vs % deleted, `Fig-3` MIA AUC comparison.
- `Table-1`: Layout K×S, retrained slice counts, wall-clock.

## Code Structure
```
code/
  __init__.py
  config.py        # dataclasses and CLI defaults
  data.py          # dataset loading, sharding & slicing utilities
  models.py        # CNN definition(s)
  trainer.py       # slice training loop + metric logging
  sisa.py          # orchestration: shard loop, aggregation, deletion handling
  metrics.py       # helpers for accuracy, F1, AUC, logging to CSV
  attacks.py       # membership inference implementation
  experiments.py   # wrappers for baseline/deletion/MIA experiments
  cli.py           # argparse entrypoints to run experiments
```

## Artifact Plan
- `results/`: raw CSV logs, JSON metadata, generated plots (matplotlib) and table.
- `report/report.tex` or `.md` compiled to `report.pdf` (≤4 pages) with figures + FM insights.
- `repro.md`: environment setup + commands to reproduce.
- `commands.ps1`: PowerShell script capturing exact commands to run experiments end-to-end.

## Workflow
1. Implement modular code with deterministic seeds and YAML/JSON config logging.
2. Provide CLI to run baseline, deletion, MIA experiments individually.
3. After experiments, run plotting scripts to populate required figures/table.
4. Draft report using collected results; export to PDF.

## Next Steps
- Scaffold directories/files.
- Implement dataset/model modules.
- Build SISA trainer + experiment orchestrators.
- Add plotting/reporting utilities.
- Fill `commands.ps1` and `repro.md` once code ready.
