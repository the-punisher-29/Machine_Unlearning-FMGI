from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_vs_deleted(deletion_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(deletion_csv)
    plt.figure(figsize=(5, 4))
    plt.plot(df["percent_deleted"] * 100, df["test_acc"], marker="o", label="SISA")
    plt.xlabel("% Deleted")
    plt.ylabel("Test Accuracy")
    plt.title("Fig-1: Accuracy vs % Deleted")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_time_saved_vs_deleted(deletion_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(deletion_csv)
    plt.figure(figsize=(5, 4))
    plt.bar(df["percent_deleted"] * 100, df["time_saved"], color="#4C72B0")
    plt.xlabel("% Deleted")
    plt.ylabel("Time Saved (s)")
    plt.title("Fig-2: Time Saved vs % Deleted")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_mia_auc(mia_results: Path, output_path: Path) -> None:
    results = json.loads(mia_results.read_text())
    labels: List[str] = list(results.keys())
    aucs = [results[name]["mia_auc"] for name in labels]
    plt.figure(figsize=(5, 4))
    plt.bar(labels, aucs, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylabel("AUC")
    plt.title("Fig-3: MIA AUC Comparison")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
