from __future__ import annotations

from typing import Any


_BACKEND_FACTORIES: dict[str, Any] = {}
_BACKEND_INSTANCES: dict[str, Any] = {}


def register_backend(name: str, factory: Any) -> None:
    key = _normalize_backend_name(name)
    _BACKEND_FACTORIES[key] = factory
    _BACKEND_INSTANCES.pop(key, None)


def get_backend(name: str | None = None) -> Any:
    key = _normalize_backend_name(name or "torch")
    _ensure_builtin_backends_registered()
    if key not in _BACKEND_FACTORIES:
        raise ValueError(f"unknown mlblack backend: {name!r}")
    if key not in _BACKEND_INSTANCES:
        factory = _BACKEND_FACTORIES[key]
        _BACKEND_INSTANCES[key] = factory() if callable(factory) else factory
    return _BACKEND_INSTANCES[key]


def list_backends() -> tuple[str, ...]:
    _ensure_builtin_backends_registered()
    return tuple(sorted(_BACKEND_FACTORIES))


def list_backend_capabilities(name: str | None = None) -> tuple[dict[str, Any], ...]:
    backend = get_backend(name)
    return tuple(item.as_dict() for item in backend.contract().capabilities)


def explain_backend_requirements(name: str | None, requirements: tuple[str, ...]) -> dict[str, Any]:
    backend = get_backend(name)
    contract = backend.contract()
    missing = contract.missing(tuple(str(item) for item in requirements))
    return {
        "backend": contract.name,
        "ok": not missing,
        "missing": tuple(missing),
        "requirements": tuple(str(item) for item in requirements),
        "provides": tuple(contract.provides),
    }


def resolve_backend(requirements: tuple[str, ...], *, preferred: str | None = None) -> Any:
    if preferred:
        backend = get_backend(preferred)
        missing = backend.contract().missing(tuple(str(item) for item in requirements))
        if not missing:
            return backend
        raise ValueError(
            f"backend {backend.contract().name!r} is missing required capabilities: "
            + ", ".join(str(item) for item in missing)
        )
    for name in list_backends():
        backend = get_backend(name)
        if not backend.contract().missing(tuple(str(item) for item in requirements)):
            return backend
    raise ValueError(f"no registered backend satisfies requirements: {requirements}")


def _normalize_backend_name(name: str) -> str:
    key = str(name or "torch").strip().lower()
    aliases = {
        "google_jax": "jax",
        "jax_cpu": "jax",
        "jax_neural": "jax",
        "np": "numpy",
        "numpy_cpu": "numpy",
        "numpy_neural": "numpy",
        "torch_native": "torch",
        "pytorch": "torch",
        "tf": "tensorflow",
        "tf_neural": "tensorflow",
        "tensorflow_cpu": "tensorflow",
        "tensorflow_neural": "tensorflow",
    }
    return aliases.get(key, key)


def _ensure_builtin_backends_registered() -> None:
    if {"jax", "numpy", "tensorflow", "torch"}.issubset(_BACKEND_FACTORIES):
        return
    from mlblack.backends.jax_neural.backend import JaxNeuralBackend
    from mlblack.backends.numpy_neural.backend import NumpyNeuralBackend
    from mlblack.backends.tensorflow_neural.backend import TensorFlowNeuralBackend
    from mlblack.backends.torch_neural.backend import TorchNeuralBackend

    register_backend("jax", JaxNeuralBackend)
    register_backend("numpy", NumpyNeuralBackend)
    register_backend("tensorflow", TensorFlowNeuralBackend)
    register_backend("torch", TorchNeuralBackend)


__all__ = [
    "explain_backend_requirements",
    "get_backend",
    "list_backend_capabilities",
    "list_backends",
    "register_backend",
    "resolve_backend",
]
