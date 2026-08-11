from __future__ import annotations

import numpy as np
import torch

from ecg_experiment.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    optimize_multilabel_thresholds,
)
from ecg_experiment.models import ClassicalECGClassifier


def test_classical_model_preserves_batch_and_class_dimensions() -> None:
    model = ClassicalECGClassifier(num_classes=9, in_channels=12, latent_dim=16)
    output = model(torch.randn(3, 12, 500))
    assert output.shape == (3, 9)


def test_metrics_use_probabilities_for_multiclass_scores() -> None:
    targets = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[targets] * 0.9 + 0.1 / 3
    metrics = classification_metrics(targets, probabilities, ("A", "B", "C"))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_auprc"] == 1.0


def test_multilabel_metrics_do_not_force_one_class_per_record() -> None:
    targets = np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
    probabilities = targets * 0.8 + (1 - targets) * 0.2
    metrics = classification_metrics(
        targets, probabilities, ("A", "B", "C"), multilabel=True
    )
    assert metrics["subset_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_brier"] < 0.05


def test_thresholds_are_tuned_from_validation_probabilities() -> None:
    targets = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]])
    probabilities = np.asarray([[0.35, 0.1], [0.4, 0.2], [0.2, 0.45], [0.1, 0.4]])
    thresholds = optimize_multilabel_thresholds(targets, probabilities)
    metrics = classification_metrics(
        targets, probabilities, ("A", "B"), multilabel=True, threshold=thresholds
    )
    assert metrics["macro_f1"] == 1.0


def test_grouped_bootstrap_returns_reproducible_intervals() -> None:
    targets = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]])
    probabilities = targets * 0.8 + (1 - targets) * 0.2
    kwargs = dict(
        groups=np.asarray(["p1", "p2", "p3", "p4"]),
        multilabel=True,
        iterations=20,
        seed=9,
    )
    first = bootstrap_confidence_intervals(targets, probabilities, ("A", "B"), **kwargs)
    second = bootstrap_confidence_intervals(targets, probabilities, ("A", "B"), **kwargs)
    assert first == second
    assert first["macro_f1"]["lower"] == 1.0
