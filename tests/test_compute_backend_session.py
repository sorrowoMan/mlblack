from __future__ import annotations

import pytest

from mlblack.backends import BackendContract, explain_backend_requirements, get_backend, list_backend_catalog_entries, list_backends, register_backend
from mlblack.catalog import backend_capability_matrix, render_backend_matrix_markdown
from mlblack.core import ComputeBackendSession, ComputeBackendSpec, Trainer, get_compute_backend_from_context


class _EmptyBackend:
    name = "empty_for_test"

    def contract(self) -> BackendContract:
        return BackendContract(name=self.name, capabilities=tuple())


def test_compute_backend_session_fails_fast_on_missing_capability() -> None:
    register_backend("empty_for_test", _EmptyBackend)
    session = ComputeBackendSession(ComputeBackendSpec(name="empty_for_test"))

    with pytest.raises(ValueError, match="autograd.backward"):
        session.ensure(("autograd.backward",), consumer="unit-test")


def test_trainer_context_exposes_single_compute_backend_session() -> None:
    trainer = Trainer(compute_backend=ComputeBackendSpec(name="torch", device="cpu"))
    context = trainer.build_context()

    assert context["backend.session"] is trainer.compute_backend_session
    assert context["backend.name"] == "torch"
    assert context["backend.device"] == "cpu"


def test_compute_backend_lookup_requires_trainer_session() -> None:
    with pytest.raises(ValueError, match="session is required"):
        get_compute_backend_from_context({"backend.name": "torch"}, ("tensor.device",), consumer="unit-test")


def test_numpy_backend_catalog_and_capability_boundaries() -> None:
    assert "numpy" in list_backends()
    ok = explain_backend_requirements("numpy", ("tensor.float_tensor", "neural.lowering.mlp", "loss.mse"))
    assert ok["ok"] is True

    missing = explain_backend_requirements("numpy", ("autograd.backward", "optimizer.step"))
    assert missing["ok"] is False
    assert missing["missing"] == ("autograd.backward", "optimizer.step")

    entries = list_backend_catalog_entries()
    assert any(entry["name"] == "numpy.neural_lowering" for entry in entries)


def test_jax_backend_catalog_and_functional_capability_boundaries() -> None:
    assert "jax" in list_backends()
    ok = explain_backend_requirements("jax", ("tensor.float_tensor", "neural.lowering.mlp", "autograd.functional.grad"))
    assert ok["ok"] is True

    missing = explain_backend_requirements("jax", ("autograd.backward", "optimizer.step"))
    assert missing["ok"] is False
    assert missing["missing"] == ("autograd.backward", "optimizer.step")

    entries = list_backend_catalog_entries()
    assert any(entry["name"] == "jax.autograd" for entry in entries)


def test_tensorflow_backend_catalog_and_gradient_tape_capability_boundaries() -> None:
    assert "tensorflow" in list_backends()
    ok = explain_backend_requirements(
        "tensorflow",
        ("tensor.float_tensor", "neural.lowering.mlp", "autograd.functional.grad", "optimizer.sgd_step"),
    )
    assert ok["ok"] is True

    missing = explain_backend_requirements("tensorflow", ("autograd.backward", "optimizer.step"))
    assert missing["ok"] is False
    assert missing["missing"] == ("autograd.backward", "optimizer.step")

    entries = list_backend_catalog_entries()
    assert any(entry["name"] == "tensorflow.autograd" for entry in entries)


def test_torch_backend_exposes_probabilistic_forecast_loss() -> None:
    torch = pytest.importorskip("torch")
    requirement = explain_backend_requirements("torch", ("loss.gaussian_nll",))
    assert requirement["ok"] is True

    backend = get_backend("torch")
    output = {"head_outputs": {"deepar": {"mu": torch.zeros((2, 1)), "log_sigma": torch.zeros((2, 1))}}}
    target = torch.tensor([[1.0], [-1.0]])
    loss, mean, scale = backend.losses.gaussian_nll(output, target, "deepar")

    assert loss.requires_grad is False
    assert tuple(mean.shape) == (2, 1)
    assert torch.allclose(scale, torch.ones_like(scale))


def test_backend_capability_matrix_dashboard_surface() -> None:
    matrix = backend_capability_matrix(("autograd.functional.grad", "autograd.backward"))
    rows = {row["backend"]: row for row in matrix["rows"]}
    assert rows["jax"]["supports"]["autograd.functional.grad"] is True
    assert rows["jax"]["supports"]["autograd.backward"] is False
    assert rows["torch"]["supports"]["autograd.backward"] is True
    markdown = render_backend_matrix_markdown(matrix)
    assert "mlblack backend capability matrix" in markdown
    assert "`tensorflow`" in markdown
