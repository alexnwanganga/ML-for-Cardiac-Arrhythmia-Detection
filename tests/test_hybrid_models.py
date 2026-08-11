from __future__ import annotations

import pytest
import torch

from ecg_experiment.hybrid_models import (
    ParameterMatchedMLPFeatures,
    build_comparison_model,
    parameter_report,
)


@pytest.mark.parametrize("model_name", ["linear", "matched-mlp", "hybrid-vqc", "hybrid-qcnn"])
def test_hybrid_comparison_models_support_multilabel_logits_and_gradients(model_name: str) -> None:
    model = build_comparison_model(
        model_name, num_classes=9, encoder_dim=8, n_qubits=4, quantum_depth=1
    )
    inputs = torch.randn(2, 12, 128)
    outputs = model(inputs)
    assert outputs.shape == (2, 9)
    outputs.sum().backward()
    trainable_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in trainable_gradients)


def test_matched_mlp_is_close_to_requested_quantum_parameter_budget() -> None:
    layer = ParameterMatchedMLPFeatures(n_features=4, target_parameters=24)
    actual = sum(parameter.numel() for parameter in layer.parameters())
    assert abs(actual - 24) <= 4


def test_parameter_report_separates_shared_encoder() -> None:
    model = build_comparison_model("hybrid-vqc", 9, encoder_dim=8, n_qubits=4, quantum_depth=1)
    report = parameter_report(model)
    assert report["total"] == report["encoder"] + report["non_encoder"]
    assert report["feature_layer"] == 12

