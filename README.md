# Machine Unlearning with SISA

This repository implements a full SISA (Sharded, Isolated, Sliced, Aggregated) training pipeline on CIFAR-10 using PyTorch. It supports selective retraining for deletion requests, tracks utility/efficiency metrics, and includes a confidence-based membership inference attack for privacy evaluation.

## Features
- Deterministic sharding & slicing with per-slice checkpoints.
- Selective retraining for deletion requests (1% & 5% by default).
- Naïve full-retrain baseline for comparison.
- Automated plotting + Table-1 generation.
- Reproducible unix command list (`run.slurm`).

## Quick Start
1. `pip install -r requirements.txt`
2. Run `python -m code.sanity_train --epochs 20 --optimizer adam --device cuda` to confirm the CNN/optimizer reach good CIFAR-10 accuracy on your hardware.
3. Launch the full pipeline via `commands.ps1`, adjusting `--epochs`, `--lr`, or `--optimizer` flags when needed.
4. Execute deletion + naïve-delete scenarios, run MIA, and generate artifacts/plots for the report.

See `repro.md` for detailed instructions and artifact descriptions.
