from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from ecg_experiment.data import ECGRecord, build_manifest, make_grouped_splits


def write_record(root: Path, name: str, codes: str) -> None:
    folder = root / name[:2]
    folder.mkdir(parents=True, exist_ok=True)
    header = folder / f"{name}.hea"
    header.write_text(
        f"{name} 12 500 16\n" + "\n".join(f"{name}.mat 16 1000/mV 16 0 0 0 0 L{i}" for i in range(12))
        + f"\n#Dx: {codes}\n",
        encoding="utf-8",
    )
    savemat(folder / f"{name}.mat", {"val": np.zeros((12, 16), dtype=np.int16)})


def test_manifest_keeps_one_record_and_excludes_ambiguous_targets(tmp_path: Path) -> None:
    raw = tmp_path / "WFDBRecords"
    mapping = tmp_path / "conditions.csv"
    mapping.write_text(
        "Acronym Name,Full Name,Snomed_CT\nSR,Sinus Rhythm,1\nAF,Atrial Flutter,2\n",
        encoding="utf-8",
    )
    write_record(raw, "JS00001", "1")
    write_record(raw, "JS00002", "1,2")

    records = build_manifest(raw, mapping, ("SR", "AF"), require_single_target=True)

    assert [record.record_id for record in records] == ["JS/JS00001"]
    assert records[0].target == "SR"


def test_grouped_splits_are_disjoint_and_cover_records() -> None:
    records = tuple(
        ECGRecord(
            record_id=f"record-{label}-{index}",
            patient_id=f"patient-{label}-{index}",
            record_path=Path(f"record-{label}-{index}"),
            labels=(label,),
            target=label,
        )
        for label in ("A", "B", "C")
        for index in range(20)
    )

    splits = make_grouped_splits(records, n_splits=5, seed=7)
    all_ids = [record.record_id for split in (splits.train, splits.validation, splits.test) for record in split]

    assert len(all_ids) == len(set(all_ids)) == len(records)
    for split in (splits.train, splits.validation, splits.test):
        assert set(Counter(record.target for record in split)) == {"A", "B", "C"}


def test_multilabel_grouped_splits_keep_every_record_and_label() -> None:
    records = tuple(
        ECGRecord(
            record_id=f"record-{index}",
            patient_id=f"patient-{index}",
            record_path=Path(f"record-{index}"),
            labels=(("A", "B") if index % 3 == 0 else (("B", "C") if index % 3 == 1 else ("A", "C"))),
            target="multilabel",
        )
        for index in range(60)
    )
    splits = make_grouped_splits(
        records, n_splits=5, seed=7, multilabel=True, class_names=("A", "B", "C")
    )
    all_ids = [record.record_id for split in (splits.train, splits.validation, splits.test) for record in split]
    assert len(all_ids) == len(set(all_ids)) == len(records)
    for split in (splits.train, splits.validation, splits.test):
        assert {label for record in split for label in record.labels} == {"A", "B", "C"}


def test_manifest_rejects_missing_raw_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path / "missing", tmp_path / "conditions.csv", ("SR",))
