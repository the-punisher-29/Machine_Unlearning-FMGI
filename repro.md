# Reproduction Guide

## Environment
1. Create a fresh Python 3.10+ environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. (Optional) Enable CUDA if available for faster training.

## Data Preparation
The CIFAR-10 dataset is automatically downloaded to the `data/` directory when you run any of the training scripts (e.g., `sanity_train.py` or `cli.py`).
- Ensure you have an active internet connection for the first run.
- No manual download steps are required.

## Workflow
1. **Sanity Check (optional but recommended)**
   - Run `python -m code.sanity_train --epochs 20 --optimizer adam --device cuda` to ensure the backbone model converges (>60% accuracy) before launching SISA.
2. **Baseline SISA Training**
   - Prepares CIFAR-10, performs K×S training, logs metrics, and saves checkpoints.
3. **Deletion Scenarios**
   - Reuses slice checkpoints to retrain only affected slices for specified deletion percentages (default 1% and 5%).
   - **Micro-Deletion**: Run with `--percent 0.0002` to simulate removing a very small number of samples (approx. 10), which highlights SISA's efficiency when few shards are impacted.
4. **Naïve Delete Baseline**
   - Fully retrains SISA on datasets with deletions applied for comparison.
5. **Membership Inference Attack (MIA)**
   - Evaluates privacy leakage using confidence-based AUC comparisons across baseline, naïve delete, and SISA-unlearned models.
6. **Plotting & Report**
   - Generates required figures and aggregates metrics into the final report.

## Key Outputs
- `results/baseline_summary.json`
- `results/deletion_results.csv`
- `results/mia_results.json`
- `results/figures/{fig1,fig2,fig3}.png`
- `report/report.pdf`

## Commands
All experiment commands are listed in `commands.ps1` for convenience. Execute them from the repository root using PowerShell.
