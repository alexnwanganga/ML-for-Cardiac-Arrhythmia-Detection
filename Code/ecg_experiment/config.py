from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_TARGET_CLASSES = (
    "SR",
    "AF",
    "1AVB",
    "LBBB",
    "RBBB",
    "APB",
    "VPB",
    "STDD",
    "STE",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by data preparation, training, and evaluation."""

    seed: int = 43
    task: str = "multilabel"
    sample_length: int = 5000
    expected_leads: int = 12
    batch_size: int = 64
    epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    latent_dim: int = 32
    n_splits: int = 5
    test_fold: int = 0
    validation_fold: int = 0
    num_workers: int = 0
    primary_metric: str = "macro_auprc"
    target_classes: tuple[str, ...] = DEFAULT_TARGET_CLASSES

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_classes"] = list(self.target_classes)
        return result

    def validate(self) -> None:
        if self.task not in {"multilabel", "single-label"}:
            raise ValueError("task must be 'multilabel' or 'single-label'")
        if self.sample_length <= 0 or self.expected_leads <= 0:
            raise ValueError("sample_length and expected_leads must be positive")
        if self.batch_size <= 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, epochs, and patience must be positive")
        if self.n_splits < 3:
            raise ValueError("n_splits must be at least 3")
        if not 0 <= self.test_fold < self.n_splits:
            raise ValueError("test_fold is outside the configured folds")
        if len(set(self.target_classes)) != len(self.target_classes):
            raise ValueError("target_classes contains duplicates")


def resolve_repo_root(start: str | Path | None = None) -> Path:
    path = Path(start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "References" / "ConditionNames_SNOMED-CT.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root")
