from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import compare_hybrid_predictions


def test_paired_comparison_rejects_no_alignment_and_reports_zero_for_identical_models(
    tmp_path: Path, monkeypatch
) -> None:
    rows = []
    for index in range(20):
        true_a = int(index % 2 == 0)
        true_b = 1 - true_a
        rows.append(
            {
                "record_id": f"r{index}",
                "patient_id": f"p{index}",
                "true_A": true_a,
                "true_B": true_b,
                "prob_A": 0.8 if true_a else 0.2,
                "prob_B": 0.8 if true_b else 0.2,
            }
        )
    prediction_paths = []
    for name in ("first", "second"):
        folder = tmp_path / name
        folder.mkdir()
        prediction_path = folder / "test_predictions.csv"
        pd.DataFrame(rows).to_csv(prediction_path, index=False)
        (folder / "decision_thresholds.json").write_text(
            json.dumps({"class_names": ["A", "B"], "thresholds": [0.5, 0.5]}),
            encoding="utf-8",
        )
        prediction_paths.append(prediction_path)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_hybrid_predictions.py",
            "--first",
            str(prediction_paths[0]),
            "--second",
            str(prediction_paths[1]),
            "--iterations",
            "20",
            "--output",
            str(output),
        ],
    )
    compare_hybrid_predictions.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["paired_intervals"]["macro_auprc"]["difference_first_minus_second"] == 0.0

