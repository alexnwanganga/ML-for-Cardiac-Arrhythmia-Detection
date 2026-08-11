from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    multilabel: bool = False,
    threshold: float | np.ndarray = 0.5,
) -> dict[str, Any]:
    labels = np.arange(len(class_names))
    if multilabel:
        thresholds = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (len(class_names),))
        predictions = (probabilities >= thresholds[None, :]).astype(np.int64)
        result: dict[str, Any] = {
            "subset_accuracy": float(accuracy_score(targets, predictions)),
            "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
            "macro_auprc": float(average_precision_score(targets, probabilities, average="macro")),
            "thresholds": thresholds.tolist(),
            "per_class": {},
        }
        try:
            result["macro_auroc"] = float(roc_auc_score(targets, probabilities, average="macro"))
        except ValueError:
            result["macro_auroc"] = float("nan")
        per_brier = np.mean(np.square(probabilities - targets), axis=0)
        result["macro_brier"] = float(per_brier.mean())
        result["macro_ece"] = float(
            np.mean(
                [
                    _binary_expected_calibration_error(targets[:, index], probabilities[:, index])
                    for index in range(len(class_names))
                ]
            )
        )
        per_precision = precision_score(targets, predictions, average=None, zero_division=0)
        per_recall = recall_score(targets, predictions, average=None, zero_division=0)
        per_f1 = f1_score(targets, predictions, average=None, zero_division=0)
        per_auprc = average_precision_score(targets, probabilities, average=None)
        for index, name in enumerate(class_names):
            result["per_class"][name] = {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "auprc": float(per_auprc[index]),
                "brier": float(per_brier[index]),
            }
        return result

    predictions = probabilities.argmax(axis=1)
    one_hot = label_binarize(targets, classes=labels)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(
            precision_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "per_class": {},
    }
    try:
        result["macro_auprc"] = float(average_precision_score(one_hot, probabilities, average="macro"))
        result["macro_auroc"] = float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        result["macro_auprc"] = float("nan")
        result["macro_auroc"] = float("nan")

    per_precision = precision_score(targets, predictions, labels=labels, average=None, zero_division=0)
    per_recall = recall_score(targets, predictions, labels=labels, average=None, zero_division=0)
    per_f1 = f1_score(targets, predictions, labels=labels, average=None, zero_division=0)
    for index, name in enumerate(class_names):
        result["per_class"][name] = {
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
        }
    return result


def _binary_expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(targets)
    error = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == n_bins - 1 else probabilities < upper
        )
        if np.any(mask):
            error += float(mask.mean()) * abs(float(probabilities[mask].mean() - targets[mask].mean()))
    return error if total else float("nan")


def optimize_multilabel_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Choose per-label F1 thresholds using validation predictions only."""

    candidates = np.asarray(grid if grid is not None else np.linspace(0.05, 0.95, 19))
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float64)
    for class_index in range(targets.shape[1]):
        best_score = -1.0
        best_threshold = 0.5
        for candidate in candidates:
            predictions = (probabilities[:, class_index] >= candidate).astype(np.int64)
            score = f1_score(targets[:, class_index], predictions, zero_division=0)
            if score > best_score or (
                np.isclose(score, best_score)
                and abs(float(candidate) - 0.5) < abs(best_threshold - 0.5)
            ):
                best_score = float(score)
                best_threshold = float(candidate)
        thresholds[class_index] = best_threshold
    return thresholds


def bootstrap_confidence_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    groups: Sequence[str] | np.ndarray,
    multilabel: bool,
    threshold: float | np.ndarray = 0.5,
    iterations: int = 1000,
    seed: int = 43,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Grouped nonparametric bootstrap intervals for final test metrics."""

    if iterations <= 0:
        return {}
    groups_array = np.asarray(groups)
    if len(groups_array) != len(targets):
        raise ValueError("groups and targets must have equal length")
    unique_groups = np.unique(groups_array)
    group_indices = {group: np.flatnonzero(groups_array == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    metric_names = (
        ("macro_auprc", "macro_auroc", "macro_f1", "macro_recall", "macro_brier")
        if multilabel
        else ("accuracy", "macro_auprc", "macro_auroc", "macro_f1", "macro_recall")
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sampled_targets = targets[indices]
        if multilabel:
            positives = sampled_targets.sum(axis=0)
            if np.any(positives == 0) or np.any(positives == len(sampled_targets)):
                continue
        elif len(np.unique(sampled_targets)) < len(class_names):
            continue
        metrics = classification_metrics(
            sampled_targets,
            probabilities[indices],
            class_names,
            multilabel=multilabel,
            threshold=threshold,
        )
        for name in metric_names:
            value = float(metrics[name])
            if np.isfinite(value):
                samples[name].append(value)

    alpha = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        if values:
            result[name] = {
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
                "confidence": confidence,
                "bootstrap_samples": len(values),
            }
    return result
