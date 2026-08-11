from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_experiment.config import DEFAULT_TARGET_CLASSES, ExperimentConfig, resolve_repo_root
from ecg_experiment.data import (
    ECGDataset,
    SplitRecords,
    build_manifest,
    class_weights,
    compute_channel_statistics,
    make_grouped_splits,
    seed_worker,
)
from ecg_experiment.models import ClassicalECGClassifier
from ecg_experiment.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    optimize_multilabel_thresholds,
)
from ecg_experiment.training import (
    fit_classifier,
    predict,
    save_json,
    save_prediction_table,
    select_device,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leakage-safe 12-lead CNN experiment")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/classical"))
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--task", choices=("multilabel", "single-label"), default="multilabel")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--target-classes",
        default=",".join(DEFAULT_TARGET_CLASSES),
        help="Comma-separated mutually exclusive target acronyms",
    )
    return parser.parse_args()


def write_split_manifest(path: Path, splits: SplitRecords) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "record_id", "patient_id", "record_path", "target", "labels"),
        )
        writer.writeheader()
        for split_name, records in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        ):
            for record in records:
                writer.writerow(
                    {
                        "split": split_name,
                        "record_id": record.record_id,
                        "patient_id": record.patient_id,
                        "record_path": str(record.record_path),
                        "target": record.target,
                        "labels": "|".join(record.labels),
                    }
                )


def make_loader(
    dataset: ECGDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    target_classes = tuple(item.strip() for item in args.target_classes.split(",") if item.strip())
    config = ExperimentConfig(
        seed=args.seed,
        task=args.task,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        num_workers=args.num_workers,
        target_classes=target_classes,
    )
    config.validate()
    set_global_seed(config.seed)

    records = build_manifest(
        repo_root / "Data" / "WFDBRecords",
        repo_root / "References" / "ConditionNames_SNOMED-CT.csv",
        config.target_classes,
        require_single_target=config.task == "single-label",
    )
    counts = Counter(label for record in records for label in record.labels)
    missing = set(config.target_classes) - set(counts)
    if missing:
        raise ValueError(f"No eligible raw records found for classes: {sorted(missing)}")

    splits = make_grouped_splits(
        records,
        n_splits=config.n_splits,
        test_fold=config.test_fold,
        validation_fold=config.validation_fold,
        seed=config.seed,
        multilabel=config.task == "multilabel",
        class_names=config.target_classes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output_dir / "split_manifest.csv", splits)
    save_json(output_dir / "config.json", config.to_dict())
    save_json(
        output_dir / "class_counts.json",
        {
            name: dict(Counter(label for record in split for label in record.labels))
            for name, split in (
                ("train", splits.train),
                ("validation", splits.validation),
                ("test", splits.test),
            )
        },
    )

    class_names = tuple(config.target_classes)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    channel_mean, channel_std = compute_channel_statistics(
        splits.train, config.expected_leads, config.sample_length
    )
    save_json(
        output_dir / "normalization.json",
        {"mean": channel_mean.tolist(), "std": channel_std.tolist()},
    )

    common_dataset_args = {
        "class_to_index": class_to_index,
        "expected_leads": config.expected_leads,
        "sample_length": config.sample_length,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "multilabel": config.task == "multilabel",
    }
    train_dataset = ECGDataset(splits.train, augment=args.augment, **common_dataset_args)
    validation_dataset = ECGDataset(splits.validation, augment=False, **common_dataset_args)
    test_dataset = ECGDataset(splits.test, augment=False, **common_dataset_args)

    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 2,
    )

    model = ClassicalECGClassifier(
        len(class_names),
        in_channels=config.expected_leads,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    )
    device = select_device()
    result = fit_classifier(
        model,
        train_loader,
        validation_loader,
        class_names,
        device=device,
        class_weight=class_weights(
            splits.train, class_to_index, multilabel=config.task == "multilabel"
        ),
        epochs=config.epochs,
        patience=config.patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        primary_metric=config.primary_metric,
        multilabel=config.task == "multilabel",
    )
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    save_json(
        output_dir / "training.json",
        {
            "best_epoch": result.best_epoch,
            "best_validation_metric": result.best_validation_metric,
            "history": list(result.history),
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    )

    # The held-out test set is touched once, after architecture selection and
    # early stopping have completed using training/validation only.
    multilabel = config.task == "multilabel"
    validation_targets, validation_probabilities = predict(
        model, validation_loader, device, multilabel=multilabel
    )
    save_prediction_table(
        output_dir / "validation_predictions.csv",
        splits.validation,
        validation_targets,
        validation_probabilities,
        class_names,
    )
    thresholds = (
        optimize_multilabel_thresholds(validation_targets, validation_probabilities)
        if multilabel
        else 0.5
    )
    save_json(
        output_dir / "decision_thresholds.json",
        {"class_names": list(class_names), "thresholds": np.asarray(thresholds).tolist()},
    )
    test_targets, test_probabilities = predict(model, test_loader, device, multilabel=multilabel)
    save_prediction_table(
        output_dir / "test_predictions.csv",
        splits.test,
        test_targets,
        test_probabilities,
        class_names,
    )
    test_metrics = classification_metrics(
        test_targets,
        test_probabilities,
        class_names,
        multilabel=multilabel,
        threshold=thresholds,
    )
    test_metrics["confidence_intervals"] = bootstrap_confidence_intervals(
        test_targets,
        test_probabilities,
        class_names,
        groups=[record.patient_id for record in splits.test],
        multilabel=multilabel,
        threshold=thresholds,
        iterations=args.bootstrap_iterations,
        seed=config.seed,
    )
    save_json(output_dir / "test_metrics.json", test_metrics)
    print(json.dumps(test_metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
