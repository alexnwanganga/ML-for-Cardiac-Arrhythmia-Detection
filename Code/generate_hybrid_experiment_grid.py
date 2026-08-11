from __future__ import annotations

import csv
from pathlib import Path


MODELS = ("linear", "matched-mlp", "hybrid-vqc", "hybrid-qcnn")
QUBITS = (4, 8)
DEPTHS = (1, 2)
CONFIRMATION_SEEDS = (13, 23, 33, 43, 53)


def main() -> None:
    output = Path("artifacts/hybrid/experiment_grid.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for qubits in QUBITS:
            for depth in DEPTHS:
                rows.append(
                    {
                        "phase": "screen",
                        "model": model,
                        "n_qubits": qubits,
                        "quantum_depth": depth,
                        "embedding": "angle",
                        "seed": 43,
                        "encoder_regime": "joint",
                        "status": "pending",
                    }
                )
    for seed in CONFIRMATION_SEEDS:
        rows.append(
            {
                "phase": "confirm",
                "model": "SELECT_AFTER_SCREENING",
                "n_qubits": "PRESPECIFY_AFTER_SCREENING",
                "quantum_depth": "PRESPECIFY_AFTER_SCREENING",
                "embedding": "angle-or-reupload-ablation",
                "seed": seed,
                "encoder_regime": "frozen-and-joint",
                "status": "pending",
            }
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} prespecified runs to {output}")


if __name__ == "__main__":
    main()
