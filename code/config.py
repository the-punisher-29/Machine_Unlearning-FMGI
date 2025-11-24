from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def project_root() -> Path:
    """Return workspace root relative to this file."""
    return Path(__file__).resolve().parents[1]


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    root: Path = field(default_factory=lambda: project_root() / "data")
    download: bool = True
    val_split: float = 0.1
    num_classes: int = 10


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs_per_slice: int = 10
    lr: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9
    optimizer: str = "adam"  # defaulted from SGD -> Adam because the latter matched sanity accuracy under tight compute
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    device: str = "cuda"  # falls back to cpu if unavailable
    num_workers: int = 4
    seed: int = 42
    amp: bool = False  # automatic mixed precision flag


@dataclass
class SISAConfig:
    num_shards: int = 10
    num_slices: int = 2
    shuffle_seed: int = 123
    aggregation_mode: str = "mean"  # or "vote"


@dataclass
class DeletionScenario:
    percent: float  # percent of training set to delete
    description: str


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sisa: SISAConfig = field(default_factory=SISAConfig)
    deletion_percents: List[DeletionScenario] = field(
        default_factory=lambda: [
            DeletionScenario(percent=0.0002, description="0.02% delete"),
            DeletionScenario(percent=0.01, description="1% delete"),
            DeletionScenario(percent=0.05, description="5% delete"),
        ]
    )
    output_dir: Path = field(default_factory=lambda: project_root() / "results")
    checkpoint_dir: Path = field(
        default_factory=lambda: project_root() / "results" / "checkpoints"
    )
    log_interval: int = 20


def ensure_dirs(config: ExperimentConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def resolve_device(cfg: TrainingConfig) -> str:
    import torch

    requested = cfg.device.lower()
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
