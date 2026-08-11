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
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.arange(len(class_names))
    if multilabel:
        predictions = (probabilities >= threshold).astype(np.int64)
        result: dict[str, Any] = {
            "subset_accuracy": float(accuracy_score(targets, predictions)),
            "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
            "macro_auprc": float(average_precision_score(targets, probabilities, average="macro")),
            "macro_auroc": float(roc_auc_score(targets, probabilities, average="macro")),
            "threshold": threshold,
            "per_class": {},
        }
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
