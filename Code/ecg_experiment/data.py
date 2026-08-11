from __future__ import annotations

import csv
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import wfdb
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset


DX_PATTERN = re.compile(r"^#\s*Dx\s*:\s*(.+)$", re.IGNORECASE)
CODE_PATTERN = re.compile(r"\d+")


@dataclass(frozen=True)
class ECGRecord:
    """One original ECG recording and its target labels.

    A record appears exactly once in the manifest. This prevents copied class
    folders or individual leads from crossing experiment splits.
    """

    record_id: str
    patient_id: str
    record_path: Path
    labels: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class SplitRecords:
    train: tuple[ECGRecord, ...]
    validation: tuple[ECGRecord, ...]
    test: tuple[ECGRecord, ...]


def load_snomed_mapping(path: str | Path) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = str(row["Snomed_CT"]).strip()
            acronym = str(row["Acronym Name"]).strip()
            if code and acronym:
                aliases.setdefault(code, [])
                if acronym not in aliases[code]:
                    aliases[code].append(acronym)
    return {code: tuple(names) for code, names in aliases.items()}


def parse_diagnostic_codes(header_path: str | Path) -> tuple[str, ...]:
    for line in Path(header_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = DX_PATTERN.match(line.strip())
        if match:
            return tuple(CODE_PATTERN.findall(match.group(1)))
    return ()


def build_manifest(
    raw_records_dir: str | Path,
    condition_map_path: str | Path,
    target_classes: Sequence[str],
    *,
    require_single_target: bool = True,
) -> tuple[ECGRecord, ...]:
    """Build one manifest entry per original WFDB record.

    Records are never duplicated. ``require_single_target=True`` creates a
    controlled mutually exclusive subset; ``False`` preserves co-occurring
    diagnoses for the primary multilabel experiment.
    """

    raw_dir = Path(raw_records_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw record directory does not exist: {raw_dir}")

    code_to_labels = load_snomed_mapping(condition_map_path)
    selected = set(target_classes)
    header_paths = sorted(raw_dir.rglob("*.hea"))

    def parse_header(header_path: Path) -> ECGRecord | None:
        mat_path = header_path.with_suffix(".mat")
        if not mat_path.exists():
            return None
        relative = header_path.relative_to(raw_dir).with_suffix("")
        record_id = relative.as_posix()
        labels = tuple(
            dict.fromkeys(
                code_to_label
                for code in parse_diagnostic_codes(header_path)
                for code_to_label in code_to_labels.get(code, ())
                if code_to_label in selected
            )
        )
        if not labels or (require_single_target and len(labels) != 1):
            return None

        # This PhysioNet release contains one ECG per de-identified patient.
        # Keeping patient_id explicit allows a true patient identifier to be
        # substituted later without changing the splitting API.
        return ECGRecord(
            record_id=record_id,
            patient_id=record_id,
            record_path=header_path.with_suffix(""),
            labels=labels,
            target=labels[0] if len(labels) == 1 else "|".join(labels),
        )

    if len(header_paths) > 100:
        workers = min(32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            parsed = executor.map(parse_header, header_paths, chunksize=128)
            records = [record for record in parsed if record is not None]
    else:
        records = [record for path in header_paths if (record := parse_header(path)) is not None]

    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate record id in raw record tree")

    if not records:
        raise ValueError("No eligible records were found for the selected classes")
    return tuple(records)


def _fold_indices(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    selected_fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))
    return folds[selected_fold]


def make_grouped_splits(
    records: Sequence[ECGRecord],
    *,
    n_splits: int = 5,
    test_fold: int = 0,
    validation_fold: int = 0,
    seed: int = 43,
    multilabel: bool = False,
    class_names: Sequence[str] | None = None,
) -> SplitRecords:
    """Create deterministic, stratified, non-overlapping grouped splits."""

    if multilabel:
        if not class_names:
            raise ValueError("class_names is required for multilabel splitting")
        return _make_multilabel_grouped_splits(
            records,
            class_names=class_names,
            n_splits=n_splits,
            test_fold=test_fold,
            validation_fold=validation_fold,
            seed=seed,
        )

    y = np.asarray([record.target for record in records])
    groups = np.asarray([record.patient_id for record in records])
    remaining_idx, test_idx = _fold_indices(
        y, groups, n_splits=n_splits, selected_fold=test_fold, seed=seed
    )

    remaining_y = y[remaining_idx]
    remaining_groups = groups[remaining_idx]
    inner_splits = min(n_splits, int(np.unique(remaining_groups).size))
    if inner_splits < 2:
        raise ValueError("Not enough training groups to create a validation split")
    inner_train, inner_validation = _fold_indices(
        remaining_y,
        remaining_groups,
        n_splits=inner_splits,
        selected_fold=validation_fold % inner_splits,
        seed=seed + 1,
    )
    train_idx = remaining_idx[inner_train]
    validation_idx = remaining_idx[inner_validation]

    result = SplitRecords(
        train=tuple(records[index] for index in train_idx),
        validation=tuple(records[index] for index in validation_idx),
        test=tuple(records[index] for index in test_idx),
    )
    assert_disjoint_splits(result)
    return result


def _assign_multilabel_folds(
    records: Sequence[ECGRecord],
    class_names: Sequence[str],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Greedily balance label prevalence while keeping groups intact."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(record.patient_id, []).append(index)
    if len(grouped) < n_splits:
        raise ValueError("There are fewer patient groups than folds")

    group_items: list[tuple[str, list[int], np.ndarray]] = []
    for group, indices in grouped.items():
        vector = np.zeros(len(class_names), dtype=np.float64)
        for index in indices:
            for label in records[index].labels:
                if label in class_to_index:
                    vector[class_to_index[label]] += 1
        group_items.append((group, indices, vector))

    totals = np.sum([item[2] for item in group_items], axis=0)
    if np.any(totals == 0):
        missing = [class_names[index] for index in np.flatnonzero(totals == 0)]
        raise ValueError(f"No records found for labels: {missing}")
    rarity = 1.0 / totals
    rng = np.random.default_rng(seed)
    tie_breakers = {group: float(rng.random()) for group, _, _ in group_items}
    group_items.sort(
        key=lambda item: (
            -float(np.dot(item[2] > 0, rarity)),
            -float(item[2].sum()),
            tie_breakers[item[0]],
        )
    )

    desired_labels = totals / n_splits
    desired_size = len(records) / n_splits
    fold_labels = np.zeros((n_splits, len(class_names)), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    assignments = np.full(len(records), -1, dtype=np.int64)

    for _, indices, vector in group_items:
        best_fold = 0
        best_score = -float("inf")
        for fold in range(n_splits):
            label_need = (desired_labels - fold_labels[fold]) / np.maximum(desired_labels, 1)
            label_score = float(np.dot(label_need, vector))
            size_score = float((desired_size - fold_sizes[fold]) / max(desired_size, 1))
            score = label_score + 0.25 * size_score
            if score > best_score or (
                np.isclose(score, best_score) and fold_sizes[fold] < fold_sizes[best_fold]
            ):
                best_score = score
                best_fold = fold
        fold_labels[best_fold] += vector
        fold_sizes[best_fold] += len(indices)
        assignments[indices] = best_fold

    if np.any(assignments < 0) or np.any(fold_sizes == 0):
        raise AssertionError("Multilabel fold assignment failed")
    return assignments


def _make_multilabel_grouped_splits(
    records: Sequence[ECGRecord],
    *,
    class_names: Sequence[str],
    n_splits: int,
    test_fold: int,
    validation_fold: int,
    seed: int,
) -> SplitRecords:
    outer_assignments = _assign_multilabel_folds(records, class_names, n_splits, seed)
    test_idx = np.flatnonzero(outer_assignments == test_fold)
    remaining_idx = np.flatnonzero(outer_assignments != test_fold)
    remaining = tuple(records[index] for index in remaining_idx)
    inner_splits = min(n_splits, len({record.patient_id for record in remaining}))
    inner_assignments = _assign_multilabel_folds(remaining, class_names, inner_splits, seed + 1)
    validation_inner = np.flatnonzero(inner_assignments == validation_fold % inner_splits)
    train_inner = np.flatnonzero(inner_assignments != validation_fold % inner_splits)
    result = SplitRecords(
        train=tuple(remaining[index] for index in train_inner),
        validation=tuple(remaining[index] for index in validation_inner),
        test=tuple(records[index] for index in test_idx),
    )
    assert_disjoint_splits(result)
    return result


def assert_disjoint_splits(splits: SplitRecords) -> None:
    group_sets = {
        name: {record.patient_id for record in values}
        for name, values in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
    }
    record_sets = {
        name: {record.record_id for record in values}
        for name, values in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
    }
    names = tuple(group_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if group_sets[left] & group_sets[right]:
                raise AssertionError(f"Patient leakage between {left} and {right}")
            if record_sets[left] & record_sets[right]:
                raise AssertionError(f"Record leakage between {left} and {right}")


def load_signal(record_path: str | Path, expected_leads: int, sample_length: int) -> np.ndarray:
    record = wfdb.rdrecord(str(record_path))
    signal = np.asarray(record.p_signal, dtype=np.float32).T
    if signal.ndim != 2 or signal.shape[0] != expected_leads:
        raise ValueError(
            f"{record_path} has shape {signal.shape}; expected ({expected_leads}, time)"
        )
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if signal.shape[1] >= sample_length:
        return signal[:, :sample_length]
    padded = np.zeros((expected_leads, sample_length), dtype=np.float32)
    padded[:, : signal.shape[1]] = signal
    return padded


def compute_channel_statistics(
    records: Sequence[ECGRecord], expected_leads: int, sample_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-lead normalization statistics using training records only."""

    total = np.zeros(expected_leads, dtype=np.float64)
    total_sq = np.zeros(expected_leads, dtype=np.float64)
    count = 0
    for record in records:
        signal = load_signal(record.record_path, expected_leads, sample_length)
        total += signal.sum(axis=1, dtype=np.float64)
        total_sq += np.square(signal, dtype=np.float64).sum(axis=1)
        count += signal.shape[1]
    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class ECGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: Sequence[ECGRecord],
        class_to_index: dict[str, int],
        *,
        expected_leads: int = 12,
        sample_length: int = 5000,
        channel_mean: np.ndarray | None = None,
        channel_std: np.ndarray | None = None,
        augment: bool = False,
        multilabel: bool = False,
    ) -> None:
        self.records = tuple(records)
        self.class_to_index = dict(class_to_index)
        self.expected_leads = expected_leads
        self.sample_length = sample_length
        self.channel_mean = None if channel_mean is None else np.asarray(channel_mean, dtype=np.float32)
        self.channel_std = None if channel_std is None else np.asarray(channel_std, dtype=np.float32)
        self.augment = augment
        self.multilabel = multilabel
        if (self.channel_mean is None) != (self.channel_std is None):
            raise ValueError("channel_mean and channel_std must be provided together")

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, signal: torch.Tensor) -> torch.Tensor:
        # Conservative waveform augmentations; training split only.
        if torch.rand(()) < 0.5:
            signal = signal * torch.empty(()).uniform_(0.9, 1.1)
        if torch.rand(()) < 0.5:
            shift = int(torch.randint(-50, 51, ()).item())
            signal = torch.roll(signal, shifts=shift, dims=-1)
        if torch.rand(()) < 0.3:
            signal = signal + torch.randn_like(signal) * 0.01
        if torch.rand(()) < 0.1:
            lead = int(torch.randint(0, signal.shape[0], ()).item())
            signal[lead] = 0
        return signal

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.records[index]
        signal = load_signal(item.record_path, self.expected_leads, self.sample_length)
        if self.channel_mean is not None and self.channel_std is not None:
            signal = (signal - self.channel_mean[:, None]) / self.channel_std[:, None]
        tensor = torch.from_numpy(signal.copy())
        if self.augment:
            tensor = self._augment(tensor)
        if self.multilabel:
            target = torch.zeros(len(self.class_to_index), dtype=torch.float32)
            for label in item.labels:
                if label in self.class_to_index:
                    target[self.class_to_index[label]] = 1.0
        else:
            target = torch.tensor(self.class_to_index[item.target], dtype=torch.long)
        return tensor, target


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def class_weights(
    records: Iterable[ECGRecord], class_to_index: dict[str, int], *, multilabel: bool = False
) -> torch.Tensor:
    records = tuple(records)
    counts = np.zeros(len(class_to_index), dtype=np.float64)
    for record in records:
        labels = record.labels if multilabel else (record.target,)
        for label in labels:
            if label in class_to_index:
                counts[class_to_index[label]] += 1
    if np.any(counts == 0):
        missing = [name for name, index in class_to_index.items() if counts[index] == 0]
        raise ValueError(f"Training split has no samples for classes: {missing}")
    if multilabel:
        # BCEWithLogitsLoss pos_weight: negatives / positives per label.
        weights = (len(records) - counts) / counts
    else:
        weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
