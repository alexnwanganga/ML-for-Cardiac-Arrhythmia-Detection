from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import classification_metrics


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_metric: float


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, *, multilabel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probability = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=1)
            probabilities.append(probability.cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_names: Sequence[str],
    *,
    device: torch.device,
    class_weight: torch.Tensor,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    primary_metric: str = "macro_auprc",
    multilabel: bool = False,
) -> TrainingResult:
    model.to(device)
    criterion: nn.Module
    if multilabel:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight.to(device))
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            label_dtype = torch.float32 if multilabel else torch.long
            labels = labels.to(device, dtype=label_dtype)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * labels.shape[0]
            examples += labels.shape[0]

        validation_targets, validation_probabilities = predict(
            model, validation_loader, device, multilabel=multilabel
        )
        metrics = classification_metrics(
            validation_targets, validation_probabilities, class_names, multilabel=multilabel
        )
        score = float(metrics[primary_metric])
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(examples, 1),
            "validation_macro_auprc": float(metrics["macro_auprc"]),
            "validation_macro_f1": float(metrics["macro_f1"]),
        }
        history.append(epoch_result)

        if np.isfinite(score) and score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(tuple(history), best_epoch, best_metric)


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    class_names: Sequence[str],
    device: torch.device,
    *,
    multilabel: bool = False,
) -> dict[str, Any]:
    targets, probabilities = predict(model, loader, device, multilabel=multilabel)
    return classification_metrics(targets, probabilities, class_names, multilabel=multilabel)


def save_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_prediction_table(
    path: str | Path,
    records: Sequence[Any],
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Save record-aligned targets and probabilities for audit/paired tests."""

    if len(records) != len(targets) or len(records) != len(probabilities):
        raise ValueError("Records, targets, and probabilities must have equal length")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    multilabel = targets.ndim == 2
    fieldnames = ["record_id", "patient_id"]
    if multilabel:
        fieldnames.extend(f"true_{name}" for name in class_names)
    else:
        fieldnames.append("true_class_index")
    fieldnames.extend(f"prob_{name}" for name in class_names)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            row: dict[str, Any] = {
                "record_id": record.record_id,
                "patient_id": record.patient_id,
            }
            if multilabel:
                row.update(
                    {f"true_{name}": int(targets[index, class_index]) for class_index, name in enumerate(class_names)}
                )
            else:
                row["true_class_index"] = int(targets[index])
            row.update(
                {f"prob_{name}": float(probabilities[index, class_index]) for class_index, name in enumerate(class_names)}
            )
            writer.writerow(row)
