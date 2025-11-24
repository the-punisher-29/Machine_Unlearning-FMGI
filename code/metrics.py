from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


@dataclass
class RunningAverage:
    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


@dataclass
class Timer:
    start_time: float = field(default_factory=time.time)

    def restart(self) -> None:
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time


class CSVLogger:
    def __init__(self, path: Path, headers: Iterable[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = list(headers)
        if not self.path.exists():
            with self.path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.headers)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.headers)
            writer.writerow(row)


@torch.no_grad()
def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, float]:
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == labels).float().mean().item()
    f1 = f1_score(preds, labels)
    return accuracy, f1


def f1_score(preds: torch.Tensor, labels: torch.Tensor, epsilon: float = 1e-9) -> float:
    num_classes = int(labels.max().item() + 1)
    f1_sum = 0.0
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).float().sum().item()
        fp = ((preds == c) & (labels != c)).float().sum().item()
        fn = ((preds != c) & (labels == c)).float().sum().item()
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1_sum += f1
    return f1_sum / num_classes
