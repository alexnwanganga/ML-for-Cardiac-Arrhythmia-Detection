from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ecg_experiment.config import DEFAULT_TARGET_CLASSES, ExperimentConfig, resolve_repo_root
from ecg_experiment.data import (
    ECGDataset,
    build_manifest,
    class_weights,
    compute_channel_statistics,
    make_grouped_splits,
)
from ecg_experiment.hybrid_models import (
    build_comparison_model,
    freeze_encoder,
    load_pretrained_encoder,
    parameter_report,
)
from ecg_experiment.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    optimize_multilabel_thresholds,
)
from ecg_experiment.training import (
    fit_classifier,
    predict,
    save_json,
    select_device,
    set_global_seed,
)
from run_classical_experiment import make_loader, write_split_manifest


MODEL_CHOICES = ("linear", "matched-mlp", "hybrid-vqc", "hybrid-qcnn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled classical/hybrid ECG comparison")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--stage", choices=("validate", "test"), default="validate")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-dim", type=int, default=32)
    parser.add_argument("--n-qubits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--quantum-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--quantum-device", default="default.qubit")
    parser.add_argument("--embedding", choices=("angle", "reupload"), default="angle")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--noise-probability", type=float, default=0.0)
    parser.add_argument("--matched-to", choices=("vqc", "qcnn"), default="vqc")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def model_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.model,
        "seed": args.seed,
        "encoder_dim": args.encoder_dim,
        "n_qubits": args.n_qubits,
        "quantum_depth": args.quantum_depth,
        "quantum_device": args.quantum_device,
        "embedding": args.embedding,
        "shots": args.shots,
        "noise_probability": args.noise_probability,
        "matched_to": args.matched_to,
        "freeze_encoder": args.freeze_encoder,
        "encoder_checkpoint": None if args.encoder_checkpoint is None else str(args.encoder_checkpoint),
    }


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    output_dir = args.output_dir or Path(
        f"artifacts/hybrid/{args.model}/q{args.n_qubits}-d{args.quantum_depth}/seed-{args.seed}"
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    if args.stage == "validate" and checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run already exists at {output_dir}; use a new directory or pass --overwrite"
        )
    if args.stage == "validate" and args.freeze_encoder and args.encoder_checkpoint is None:
        raise ValueError("--freeze-encoder requires --encoder-checkpoint")

    config = ExperimentConfig(
        seed=args.seed,
        task="multilabel",
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.encoder_dim,
        num_workers=args.num_workers,
        target_classes=DEFAULT_TARGET_CLASSES,
    )
    config.validate()
    set_global_seed(config.seed)

    records = build_manifest(
        repo_root / "Data" / "WFDBRecords",
        repo_root / "References" / "ConditionNames_SNOMED-CT.csv",
        config.target_classes,
        require_single_target=False,
    )
    splits = make_grouped_splits(
        records,
        n_splits=config.n_splits,
        test_fold=config.test_fold,
        validation_fold=config.validation_fold,
        seed=config.seed,
        multilabel=True,
        class_names=config.target_classes,
    )
    write_split_manifest(output_dir / "split_manifest.csv", splits)
    current_model_config = model_configuration(args)
    model_config_path = output_dir / "model_config.json"
    if args.stage == "test":
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model configuration not found: {model_config_path}")
        stored_model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "model",
            "seed",
            "encoder_dim",
            "n_qubits",
            "quantum_depth",
            "quantum_device",
            "embedding",
            "shots",
            "noise_probability",
            "matched_to",
            "freeze_encoder",
        )
        mismatches = {
            key: (stored_model_config.get(key), current_model_config.get(key))
            for key in comparable_keys
            if stored_model_config.get(key) != current_model_config.get(key)
        }
        if mismatches:
            raise ValueError(f"Test configuration does not match validation run: {mismatches}")
    else:
        save_json(output_dir / "experiment_config.json", config.to_dict())
        save_json(model_config_path, current_model_config)
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

    normalization_path = output_dir / "normalization.json"
    if normalization_path.exists():
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        channel_mean = torch.tensor(normalization["mean"]).numpy()
        channel_std = torch.tensor(normalization["std"]).numpy()
    else:
        channel_mean, channel_std = compute_channel_statistics(
            splits.train, config.expected_leads, config.sample_length
        )
        save_json(
            normalization_path, {"mean": channel_mean.tolist(), "std": channel_std.tolist()}
        )

    class_names = tuple(config.target_classes)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    dataset_args = {
        "class_to_index": class_to_index,
        "expected_leads": config.expected_leads,
        "sample_length": config.sample_length,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "multilabel": True,
    }
    train_dataset = ECGDataset(splits.train, augment=args.augment, **dataset_args)
    validation_dataset = ECGDataset(splits.validation, augment=False, **dataset_args)
    test_dataset = ECGDataset(splits.test, augment=False, **dataset_args)
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

    model = build_comparison_model(
        args.model,
        len(class_names),
        encoder_dim=args.encoder_dim,
        n_qubits=args.n_qubits,
        quantum_depth=args.quantum_depth,
        dropout=config.dropout,
        quantum_device=args.quantum_device,
        matched_to=args.matched_to,
        embedding=args.embedding,
        shots=args.shots,
        noise_probability=args.noise_probability,
    )
    if args.stage == "validate" and args.encoder_checkpoint is not None:
        load_pretrained_encoder(model, args.encoder_checkpoint)
    if args.freeze_encoder:
        freeze_encoder(model, True)

    device = select_device()
    if args.stage == "test":
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Validation checkpoint not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        model.to(device)
        threshold_path = output_dir / "decision_thresholds.json"
        if not threshold_path.exists():
            raise FileNotFoundError(f"Validation thresholds not found: {threshold_path}")
        threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        if threshold_data["class_names"] != list(class_names):
            raise ValueError("Saved decision-threshold class order does not match this run")
        thresholds = np.asarray(threshold_data["thresholds"], dtype=np.float64)
        test_targets, test_probabilities = predict(model, test_loader, device, multilabel=True)
        metrics = classification_metrics(
            test_targets,
            test_probabilities,
            class_names,
            multilabel=True,
            threshold=thresholds,
        )
        metrics["confidence_intervals"] = bootstrap_confidence_intervals(
            test_targets,
            test_probabilities,
            class_names,
            groups=[record.patient_id for record in splits.test],
            multilabel=True,
            threshold=thresholds,
            iterations=args.bootstrap_iterations,
            seed=config.seed,
        )
        save_json(output_dir / "test_metrics.json", metrics)
        print(json.dumps(metrics, indent=2))
        return 0

    started = time.perf_counter()
    result = fit_classifier(
        model,
        train_loader,
        validation_loader,
        class_names,
        device=device,
        class_weight=class_weights(splits.train, class_to_index, multilabel=True),
        epochs=config.epochs,
        patience=config.patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        primary_metric=config.primary_metric,
        multilabel=True,
    )
    elapsed = time.perf_counter() - started
    torch.save(model.state_dict(), checkpoint_path)
    validation_targets, validation_probabilities = predict(
        model, validation_loader, device, multilabel=True
    )
    thresholds = optimize_multilabel_thresholds(validation_targets, validation_probabilities)
    save_json(
        output_dir / "decision_thresholds.json",
        {"class_names": list(class_names), "thresholds": thresholds.tolist()},
    )
    validation_metrics = classification_metrics(
        validation_targets,
        validation_probabilities,
        class_names,
        multilabel=True,
        threshold=thresholds,
    )
    save_json(output_dir / "validation_metrics.json", validation_metrics)
    save_json(
        output_dir / "training.json",
        {
            "best_epoch": result.best_epoch,
            "best_validation_metric": result.best_validation_metric,
            "history": list(result.history),
            "elapsed_seconds": elapsed,
            "device": str(device),
            "parameters": parameter_report(model),
        },
    )
    print(json.dumps(validation_metrics, indent=2))
    print("Test set was not evaluated. Run --stage test only for the prespecified winner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
