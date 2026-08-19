from __future__ import annotations

import pytest

from mlblack.backends import BackendContract, explain_backend_requirements, get_backend, list_backend_catalog_entries, list_backends, register_backend
from mlblack.catalog import backend_capability_matrix, render_backend_matrix_markdown
from mlblack.core import ComputeBackendSession, ComputeBackendSpec, Feedback, LearningProblem, ModelRepresentation, UnknownState, get_compute_backend_from_context
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import FixedCandidateAdapter


class _EmptyBackend:
    name = "empty_for_test"

    def contract(self) -> BackendContract:
        return BackendContract(name=self.name, capabilities=tuple())


class _Representation(ModelRepresentation):
    def init(self, context):
        del context
        return UnknownState([0.0])

    def decode(self, state, context=None):
        del context
        return state.as_array()


class _Problem(LearningProblem):
    def evaluate(self, model, state, context=None):
        del model, state, context
        return Feedback(objectives=[0.0])


def test_compute_backend_session_fails_fast_on_missing_capability() -> None:
    register_backend("empty_for_test", _EmptyBackend)
    session = ComputeBackendSession(ComputeBackendSpec(name="empty_for_test"))
    with pytest.raises(ValueError, match="autograd.backward"):
        session.ensure(("autograd.backward",), consumer="unit-test")


def test_learning_solver_context_exposes_single_compute_backend_session() -> None:
    solver = build_learning_solver(
        problem=_Problem(), representation=_Representation(),
        adapter=FixedCandidateAdapter(),
        compute_backend=ComputeBackendSpec(name="torch", device="cpu"),
    )
    context = solver.build_context()
    assert context["backend.session"] is solver.compute_backend_session
    assert context["backend.name"] == "torch"
    assert context["backend.device"] == "cpu"


def test_compute_backend_lookup_requires_solver_session() -> None:
    with pytest.raises(ValueError, match="session is required"):
        get_compute_backend_from_context({"backend.name": "torch"}, ("tensor.device",), consumer="unit-test")


def test_backend_catalog_and_capability_boundaries() -> None:
    assert {"numpy", "jax", "tensorflow", "torch"}.issubset(set(list_backends()))
    assert explain_backend_requirements("numpy", ("neural.lowering.mlp",))["ok"] is True
    assert explain_backend_requirements("jax", ("autograd.functional.grad",))["ok"] is True
    assert explain_backend_requirements("jax", ("autograd.backward",))["ok"] is False
    assert explain_backend_requirements("tensorflow", ("autograd.functional.grad",))["ok"] is True
    entries = list_backend_catalog_entries()
    assert any(entry["name"] == "jax.autograd" for entry in entries)


def test_torch_backend_exposes_probabilistic_forecast_loss() -> None:
    torch = pytest.importorskip("torch")
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
    assert rows["torch"]["supports"]["autograd.backward"] is True
    assert "`tensorflow`" in render_backend_matrix_markdown(matrix)
