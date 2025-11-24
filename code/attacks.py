from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader


def _collect_confidences(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> List[float]:
    scores: List[float] = []
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = softmax(logits)
            max_conf, _ = probs.max(dim=1)
            scores.extend(max_conf.detach().cpu().tolist())
    return scores


def _roc_auc(member_scores: Sequence[float], nonmember_scores: Sequence[float]) -> float:
    labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
    scores = list(member_scores) + list(nonmember_scores)
    order = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0.0
    fp = 0.0
    prev_score = None
    tps: List[float] = []
    fps: List[float] = []
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    for score, label in order:
        if prev_score is not None and score != prev_score:
            tps.append(tp / total_pos)
            fps.append(fp / total_neg)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    tps.append(tp / total_pos)
    fps.append(fp / total_neg)
    auc = 0.0
    for i in range(1, len(tps)):
        auc += (fps[i] - fps[i - 1]) * (tps[i] + tps[i - 1]) / 2
    return abs(auc)


def run_confidence_mia(
    model: torch.nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device_obj)
    member_scores = _collect_confidences(model, member_loader, device_obj)
    nonmember_scores = _collect_confidences(model, nonmember_loader, device_obj)
    auc = _roc_auc(member_scores, nonmember_scores)
    return {
        "mia_auc": auc,
        "member_mean_conf": float(sum(member_scores) / max(len(member_scores), 1)),
        "nonmember_mean_conf": float(
            sum(nonmember_scores) / max(len(nonmember_scores), 1)
        ),
    }
