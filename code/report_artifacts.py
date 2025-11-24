from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .plotting import (
    plot_accuracy_vs_deleted,
    plot_mia_auc,
    plot_time_saved_vs_deleted,
)


def build_table(deletion_csv: Path, baseline_json: Path, output_path: Path) -> None:
    df = pd.read_csv(deletion_csv)
    baseline = json.loads(baseline_json.read_text())
    total_slices = baseline["num_shards"] * baseline["num_slices"]
    table = pd.DataFrame(
        {
            "percent_deleted": df["percent_deleted"] * 100,
            "K": baseline["num_shards"],
            "S": baseline["num_slices"],
            "retrained_slices": df["num_retrained"],
            "total_slices": total_slices,
            "fraction_retrained": df["fraction_retrained"],
            "baseline_time": baseline["baseline_time"],
            "retrain_time": df["retrain_time"],
            "time_saved": df["time_saved"],
        }
    )
    table.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate results artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--table-path", default="results/table1.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    table_path = Path(args.table_path)
    deletion_csv = results_dir / "deletion_results.csv"
    baseline_json = results_dir / "baseline_summary.json"
    mia_json = results_dir / "mia_results.json"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_vs_deleted(deletion_csv, figures_dir / "fig1_accuracy_vs_deleted.png")
    plot_time_saved_vs_deleted(deletion_csv, figures_dir / "fig2_time_saved.png")
    plot_mia_auc(mia_json, figures_dir / "fig3_mia_auc.png")
    build_table(deletion_csv, baseline_json, table_path)


if __name__ == "__main__":
    main()
