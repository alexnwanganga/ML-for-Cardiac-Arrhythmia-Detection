from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment.metrics import classification_metrics
from ecg_experiment.training import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired grouped-bootstrap comparison of two ECG models")
    parser.add_argument("--first", type=Path, required=True, help="First test_predictions.csv")
    parser.add_argument("--second", type=Path, required=True, help="Second test_predictions.csv")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, default=Path("paired_comparison.json"))
    return parser.parse_args()


def load_thresholds(prediction_path: Path, class_names: list[str]) -> np.ndarray:
    path = prediction_path.with_name("decision_thresholds.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["class_names"] != class_names:
        raise ValueError(f"Threshold class order mismatch in {path}")
    return np.asarray(payload["thresholds"], dtype=np.float64)


def main() -> None:
    args = parse_args()
    first = pd.read_csv(args.first).sort_values("record_id").reset_index(drop=True)
    second = pd.read_csv(args.second).sort_values("record_id").reset_index(drop=True)
    identity_columns = ["record_id", "patient_id"]
    if not first[identity_columns].equals(second[identity_columns]):
        raise ValueError("Prediction tables do not contain the same records in the same groups")
    class_names = [column.removeprefix("prob_") for column in first.columns if column.startswith("prob_")]
    if not class_names:
        raise ValueError("No probability columns found")
    true_columns = [f"true_{name}" for name in class_names]
    probability_columns = [f"prob_{name}" for name in class_names]
    if not first[true_columns].equals(second[true_columns]):
        raise ValueError("Prediction tables disagree on ground-truth labels")
    targets = first[true_columns].to_numpy(dtype=np.int64)
    first_probabilities = first[probability_columns].to_numpy(dtype=np.float64)
    second_probabilities = second[probability_columns].to_numpy(dtype=np.float64)
    first_thresholds = load_thresholds(args.first, class_names)
    second_thresholds = load_thresholds(args.second, class_names)

    first_metrics = classification_metrics(
        targets, first_probabilities, class_names, multilabel=True, threshold=first_thresholds
    )
    second_metrics = classification_metrics(
        targets, second_probabilities, class_names, multilabel=True, threshold=second_thresholds
    )
    metric_names = ("macro_auprc", "macro_auroc", "macro_f1", "macro_recall", "macro_brier")
    point_differences = {
        name: float(first_metrics[name] - second_metrics[name]) for name in metric_names
    }

    groups = first["patient_id"].to_numpy()
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(args.seed)
    differences = {name: [] for name in metric_names}
    for _ in range(args.iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        positives = targets[indices].sum(axis=0)
        if np.any(positives == 0) or np.any(positives == len(indices)):
            continue
        first_sample = classification_metrics(
            targets[indices],
            first_probabilities[indices],
            class_names,
            multilabel=True,
            threshold=first_thresholds,
        )
        second_sample = classification_metrics(
            targets[indices],
            second_probabilities[indices],
            class_names,
            multilabel=True,
            threshold=second_thresholds,
        )
        for name in metric_names:
            differences[name].append(float(first_sample[name] - second_sample[name]))

    intervals = {}
    for name, values in differences.items():
        array = np.asarray(values)
        if array.size == 0:
            raise ValueError("No valid bootstrap resamples contained every target label")
        first_better = array < 0 if name == "macro_brier" else array > 0
        intervals[name] = {
            "difference_first_minus_second": point_differences[name],
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
            "probability_first_better": float(np.mean(first_better)),
            "bootstrap_samples": int(len(array)),
        }
    save_json(
        args.output,
        {
            "first": str(args.first),
            "second": str(args.second),
            "record_count": len(first),
            "first_metrics": {name: first_metrics[name] for name in metric_names},
            "second_metrics": {name: second_metrics[name] for name in metric_names},
            "paired_intervals": intervals,
        },
    )
    print(json.dumps(intervals, indent=2))


if __name__ == "__main__":
    main()
