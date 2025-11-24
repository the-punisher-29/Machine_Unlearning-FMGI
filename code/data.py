from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import DatasetConfig, TrainingConfig


@dataclass
class SliceSpec:
    shard_id: int
    slice_id: int
    indices: np.ndarray

    @property
    def tag(self) -> str:
        return f"shard{self.shard_id:02d}_slice{self.slice_id:02d}"


class CIFAR10WithIndices(datasets.CIFAR10):
    """CIFAR10 dataset that returns (image, label, idx)."""

    def __getitem__(self, index):  # type: ignore[override]
        img, target = super().__getitem__(index)
        return img, target, index


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )


def load_datasets(cfg: DatasetConfig) -> Tuple[Dataset, Dataset]:
    train_set = CIFAR10WithIndices(
        root=cfg.root,
        train=True,
        transform=get_transforms(train=True),
        download=cfg.download,
    )
    test_set = CIFAR10WithIndices(
        root=cfg.root,
        train=False,
        transform=get_transforms(train=False),
        download=cfg.download,
    )
    return train_set, test_set


def split_train_val(dataset: Dataset, val_split: float, seed: int) -> Tuple[Subset, Subset]:
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(num_samples * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_sisa_partitions(
    num_samples: int, num_shards: int, num_slices: int, seed: int
) -> List[SliceSpec]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    shards = np.array_split(indices, num_shards)
    slices: List[SliceSpec] = []
    for shard_id, shard_indices in enumerate(shards):
        shard_slices = np.array_split(shard_indices, num_slices)
        for slice_id, slice_indices in enumerate(shard_slices):
            slices.append(
                SliceSpec(
                    shard_id=shard_id,
                    slice_id=slice_id,
                    indices=slice_indices.astype(int),
                )
            )
    return slices


def slice_dataloader(
    dataset: Dataset,
    slice_spec: SliceSpec,
    train_cfg: TrainingConfig,
    shuffle: bool = True,
) -> DataLoader:
    subset = Subset(dataset, slice_spec.indices.tolist())
    return DataLoader(
        subset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )


def make_eval_loader(
    dataset: Dataset,
    train_cfg: TrainingConfig,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )


def build_index_mapping(slices: Sequence[SliceSpec]) -> Dict[int, Tuple[int, int]]:
    mapping: Dict[int, Tuple[int, int]] = {}
    for spec in slices:
        for idx in spec.indices:
            mapping[int(idx)] = (spec.shard_id, spec.slice_id)
    return mapping
