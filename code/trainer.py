from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import amp, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainingConfig
from .metrics import CSVLogger, RunningAverage, Timer, classification_metrics


class SliceTrainer:
    def __init__(self, cfg: TrainingConfig, log_interval: int = 20):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        device_type = self.device.type
        use_amp = cfg.amp and device_type == "cuda"
        self.scaler = amp.GradScaler(enabled=use_amp)
        self.log_interval = log_interval

    def _memory_snapshot(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        return 0.0

    def train_slice(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        slice_tag: str,
        log_path: Path,
    ) -> Dict[str, float]:
        model.to(self.device)
        model.train()
        logger = CSVLogger(
            path=log_path,
            headers=[
                "slice",
                "epoch",
                "loss",
                "acc",
                "f1",
                "lr",
                "wall_clock",
                "gpu_mem_gb",
            ],
        )
        timer = Timer()
        num_steps = len(train_loader)
        for epoch in range(1, epochs + 1):
            loss_meter = RunningAverage(name="loss")
            progress = tqdm(
                train_loader,
                total=num_steps,
                desc=f"{slice_tag} | epoch {epoch}/{epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            for step, batch in enumerate(progress, start=1):
                inputs, targets, _ = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(
                    device_type=self.device.type, enabled=self.scaler.is_enabled()
                ):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                loss_meter.update(loss.item(), inputs.size(0))

                progress.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

                if self.log_interval and (step % self.log_interval == 0 or step == num_steps):
                    acc, f1 = classification_metrics(outputs.detach(), targets)
                    logger.log(
                        {
                            "slice": slice_tag,
                            "epoch": epoch,
                            "loss": loss_meter.avg,
                            "acc": acc,
                            "f1": f1,
                            "lr": optimizer.param_groups[0]["lr"],
                            "wall_clock": timer.elapsed(),
                            "gpu_mem_gb": self._memory_snapshot(),
                        }
                    )
            if scheduler is not None:
                scheduler.step()

        eval_metrics = {}
        if val_loader is not None:
            eval_metrics = self.evaluate(model, val_loader)
            eval_metrics["slice"] = slice_tag
        eval_metrics["train_loss"] = loss_meter.avg
        eval_metrics["wall_clock"] = timer.elapsed()
        eval_metrics["gpu_mem_gb"] = self._memory_snapshot()
        return eval_metrics

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, data_loader: DataLoader
    ) -> Dict[str, float]:
        model.eval()
        model.to(self.device)
        acc_meter = RunningAverage("acc")
        loss_meter = RunningAverage("loss")
        f1_meter = RunningAverage("f1")
        criterion = nn.CrossEntropyLoss()
        for inputs, targets, _ in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc, f1 = classification_metrics(outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))
            f1_meter.update(f1, inputs.size(0))
        return {
            "loss": loss_meter.avg,
            "acc": acc_meter.avg,
            "f1": f1_meter.avg,
        }
