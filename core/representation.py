"""Model representation for mlblack trainers.

Inherits the unified RepresentationBase from blackbase and adds mlblack-specific
features (UnknownState type, ContractMixin, batch variants).
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from blackbase.abc import RepresentationBase

from blackbase.contracts import ComponentContract, ContractMixin
from .types import UnknownState


class ModelRepresentation(RepresentationBase, ContractMixin):
    """Unknown-state encoder/decoder, equivalent to nsgablack Representation.

    Inherits RepresentationBase (unified interface) and ContractMixin
    (mlblack metadata protocol).
    """

    name = "model_representation"
    context_requires = ('candidate.unknown_state',)
    context_optional = ()
    context_provides = ('candidate.model',)
    context_mutates = ('candidate.repaired_state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state; provides candidate.model; mutates candidate.repaired_state.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state",),
        provides=("candidate.model",),
        mutates=("candidate.repaired_state",),
        supports_batch=True,
    )

    # --- RepresentationBase abstract methods ---

    @abstractmethod
    def init(self, context: Mapping[str, Any]) -> UnknownState:
        """Create an initial unknown state."""

    @abstractmethod
    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        """Decode unknown state into a model/function."""

    # --- Optional hooks (override RepresentationBase defaults) ---

    def encode(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        raise NotImplementedError(f"{type(self).__name__}.encode(...) is not implemented")

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        return state

    def mutate(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        return state

    def fingerprint(self, state: UnknownState) -> str:
        """Return the stable semantic identity used for feedback alignment."""

        return unknown_state_fingerprint(state)

    def equivalent(self, left: UnknownState, right: UnknownState) -> bool:
        """Return whether two states decode to the same semantic candidate."""

        return self.fingerprint(left) == self.fingerprint(right)

    # --- Batch variants ---

    def init_batch(self, n: int, context: Mapping[str, Any] | None = None) -> tuple[UnknownState, ...]:
        ctx = dict(context or {})
        return tuple(self.init(ctx) for _ in range(int(n)))

    def decode_batch(self, states: Sequence[UnknownState], context: Mapping[str, Any] | None = None) -> tuple[Any, ...]:
        ctx = dict(context or {})
        return tuple(self.decode(state, ctx) for state in tuple(states))

    def repair_batch(self, states: Sequence[UnknownState], context: Mapping[str, Any] | None = None) -> tuple[UnknownState, ...]:
        ctx = dict(context or {})
        return tuple(self.repair(state, ctx) for state in tuple(states))

    # --- Override: describe uses ContractMixin ---

    def describe(self) -> Mapping[str, Any]:
        return {"name": self.name, "contract": self.get_context_contract()}


RepresentationPipeline = ModelRepresentation


def unknown_state_fingerprint(state: UnknownState) -> str:
    """Fingerprint numeric values and all decode-relevant metadata by default."""

    values = np.ascontiguousarray(state.as_array())
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    metadata = _canonical_metadata(state.metadata)
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _canonical_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, np.generic):
        return _canonical_metadata(value.item())
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": _canonical_metadata(value.tolist()),
            }
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_canonical_metadata(item) for item in value]}
    if isinstance(value, list):
        return [_canonical_metadata(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_metadata(item) for item in value]
        return {
            "__set__": sorted(
                items,
                key=lambda item: json.dumps(item, sort_keys=True, default=repr),
            )
        }
    for method_name in ("to_protocol_payload", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "payload": _canonical_metadata(method()),
            }
    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


__all__ = [
    "ModelRepresentation",
    "RepresentationPipeline",
    "unknown_state_fingerprint",
]
