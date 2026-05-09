from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


MECHANISM_RUNTIME_ORDER: dict[str, int] = {
    "state_signal_view": 0,
    "sample_weighting": 1,
    "sampling": 2,
    "aggregation": 3,
}


@dataclass(frozen=True)
class RuntimeMechanismSpec:
    key: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key or "noop").strip().lower() or "noop")
        object.__setattr__(self, "params", dict(self.params or {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": str(self.key),
            "params": dict(self.params),
        }


@dataclass
class MechanismRuntimeState:
    trainer_key: str
    family_key: str
    X: np.ndarray
    Y: np.ndarray
    feature_names: tuple[str, ...] = tuple()
    target_names: tuple[str, ...] = tuple()
    sample_weight: np.ndarray | None = None
    parent_model: Any | None = None
    parent_payload: Mapping[str, Any] | None = None
    state_signals: dict[str, Any] = field(default_factory=dict)
    row_indices: np.ndarray | None = None
    feature_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_trace(self, mechanism_key: str, *, kind: str, status: str, details: Mapping[str, Any] | None = None) -> None:
        trace = list(self.metadata.get("mechanism_trace", []))
        trace.append(
            {
                "mechanism_key": str(mechanism_key),
                "mechanism_kind": str(kind),
                "status": str(status),
                "details": {} if details is None else dict(details),
            }
        )
        self.metadata["mechanism_trace"] = trace


class RuntimeMechanismBase:
    mechanism_key = "noop"
    mechanism_kind = "sampling"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def summary(self) -> dict[str, Any]:
        return {
            "key": str(self.mechanism_key),
            "kind": str(self.mechanism_kind),
            "params": dict(self.params),
        }

    def pre_fit(self, state: MechanismRuntimeState) -> None:
        return None

    def post_fit(self, state: MechanismRuntimeState, *, model: Any, artifact_metadata: dict[str, Any]) -> None:
        return None


class PredictionResidualStateSignalView(RuntimeMechanismBase):
    mechanism_key = "state_signal_view.prediction_residual"
    mechanism_kind = "state_signal_view"

    @staticmethod
    def _apply_parent_feature_view(X: np.ndarray, parent_payload: Mapping[str, Any] | None) -> np.ndarray:
        x_in = np.asarray(X, dtype=float)
        input_feature_indices = None if parent_payload is None else parent_payload.get("input_feature_indices")
        if input_feature_indices is not None:
            idx = np.asarray(tuple(input_feature_indices), dtype=int)
            x_in = x_in[:, idx]
        return x_in

    @classmethod
    def _predict(cls, parent_model: Any, X: np.ndarray, parent_payload: Mapping[str, Any] | None) -> np.ndarray:
        x_in = cls._apply_parent_feature_view(X, parent_payload)
        pred = np.asarray(parent_model.predict(x_in), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    @staticmethod
    def _uncertainty(parent_model: Any, X: np.ndarray, parent_payload: Mapping[str, Any] | None) -> np.ndarray | None:
        estimators = tuple(getattr(parent_model, "estimators_", ()) or ())
        if len(estimators) <= 1:
            return None
        x_in = PredictionResidualStateSignalView._apply_parent_feature_view(X, parent_payload)
        estimator_features = tuple(getattr(parent_model, "estimators_features_", ()) or ())
        preds = []
        for i, est in enumerate(estimators):
            local_x = x_in
            if i < len(estimator_features):
                local_idx = np.asarray(tuple(estimator_features[i]), dtype=int)
                local_x = x_in[:, local_idx]
            y = np.asarray(est.predict(local_x), dtype=float)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            preds.append(y)
        if not preds:
            return None
        stack = np.stack(preds, axis=0)
        weights = np.asarray(getattr(parent_model, "estimator_weights_", ()), dtype=float).reshape(-1)
        if weights.size == stack.shape[0] and np.sum(weights) > 0.0:
            norm = weights / np.sum(weights)
            mean = np.tensordot(norm, stack, axes=(0, 0))
            var = np.tensordot(norm, (stack - mean) ** 2, axes=(0, 0))
            return np.sqrt(np.maximum(var, 0.0))
        return np.asarray(np.std(stack, axis=0, ddof=1), dtype=float)

    def pre_fit(self, state: MechanismRuntimeState) -> None:
        if state.parent_model is None:
            state.record_trace(self.mechanism_key, kind=self.mechanism_kind, status="skipped", details={"reason": "no_parent_model"})
            return
        pred = self._predict(state.parent_model, state.X, state.parent_payload)
        state.state_signals["prediction_ref"] = pred
        residual = np.asarray(state.Y, dtype=float) - pred
        state.state_signals["residual_ref"] = residual
        state.state_signals["per_sample_loss_ref"] = np.mean(residual ** 2, axis=1)
        unc = self._uncertainty(state.parent_model, state.X, state.parent_payload)
        if unc is not None:
            state.state_signals["uncertainty_ref"] = unc
        state.record_trace(
            self.mechanism_key,
            kind=self.mechanism_kind,
            status="applied",
            details={
                "provided_signals": sorted(str(key) for key in state.state_signals.keys()),
            },
        )


class GradientNormStateSignalView(RuntimeMechanismBase):
    mechanism_key = "state_signal_view.gradient_norm"
    mechanism_kind = "state_signal_view"

    def pre_fit(self, state: MechanismRuntimeState) -> None:
        if state.parent_model is None:
            state.record_trace(self.mechanism_key, kind=self.mechanism_kind, status="skipped", details={"reason": "no_parent_model"})
            return
        gradient_norm_fn = getattr(state.parent_model, "gradient_norm", None)
        if not callable(gradient_norm_fn):
            state.record_trace(
                self.mechanism_key,
                kind=self.mechanism_kind,
                status="skipped",
                details={"reason": "parent_gradient_norm_unsupported"},
            )
            return
        x_in = PredictionResidualStateSignalView._apply_parent_feature_view(state.X, state.parent_payload)
        strict = bool(self.params.get("strict", False))
        try:
            raw = np.asarray(gradient_norm_fn(x_in, state.Y), dtype=float)
        except Exception as exc:
            if strict:
                raise
            state.record_trace(
                self.mechanism_key,
                kind=self.mechanism_kind,
                status="skipped",
                details={"reason": "gradient_norm_failed", "error_type": type(exc).__name__},
            )
            return
        if raw.ndim > 1:
            raw = np.mean(raw, axis=1)
        grad = raw.reshape(-1)
        if grad.shape[0] != state.X.shape[0]:
            raise ValueError("gradient_norm signal length mismatch")
        state.state_signals["gradient_norm_ref"] = grad
        state.record_trace(
            self.mechanism_key,
            kind=self.mechanism_kind,
            status="applied",
            details={
                "min_gradient_norm": float(np.min(grad)),
                "max_gradient_norm": float(np.max(grad)),
            },
        )


class LossAdaptiveSampleWeightingMechanism(RuntimeMechanismBase):
    mechanism_key = "sample_weighting.loss_adaptive"
    mechanism_kind = "sample_weighting"

    def pre_fit(self, state: MechanismRuntimeState) -> None:
        source_key = str(self.params.get("source_key", "per_sample_loss_ref"))
        raw = state.state_signals.get(source_key)
        if raw is None:
            fallback_key = str(self.params.get("fallback_key", "uncertainty_ref"))
            raw = state.state_signals.get(fallback_key)
            source_key = fallback_key if raw is not None else source_key
        if raw is None:
            state.record_trace(self.mechanism_key, kind=self.mechanism_kind, status="skipped", details={"reason": "missing_signal"})
            return

        score = np.asarray(raw, dtype=float)
        if score.ndim > 1:
            score = np.mean(score, axis=1)
        score = np.maximum(score.reshape(-1), float(self.params.get("floor", 1e-8)))
        if score.shape[0] != state.X.shape[0]:
            raise ValueError("sample_weighting signal length mismatch")

        alpha = float(self.params.get("alpha", 1.0))
        power = float(self.params.get("power", 1.0))
        normalize = bool(self.params.get("normalize", True))
        multiply_existing = bool(self.params.get("multiply_existing", True))

        shaped = score ** power
        if normalize:
            shaped = shaped / max(float(np.mean(shaped)), 1e-8)
        proposed = 1.0 + alpha * shaped

        if state.sample_weight is not None and multiply_existing:
            base = np.asarray(state.sample_weight, dtype=float).reshape(-1)
            if base.shape[0] != proposed.shape[0]:
                raise ValueError("existing sample_weight length mismatch")
            out = base * proposed
        else:
            out = proposed
        out = out / max(float(np.mean(out)), 1e-8)
        state.sample_weight = np.asarray(out, dtype=float)
        state.record_trace(
            self.mechanism_key,
            kind=self.mechanism_kind,
            status="applied",
            details={
                "source_key": source_key,
                "alpha": alpha,
                "power": power,
                "min_weight": float(np.min(out)),
                "max_weight": float(np.max(out)),
            },
        )


class BatchPrioritySubsampleMechanism(RuntimeMechanismBase):
    mechanism_key = "sampling.batch_priority_subsample"
    mechanism_kind = "sampling"

    @staticmethod
    def _flatten_scores(value: Any, *, expected_len: int) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        arr = arr.reshape(-1)
        if arr.shape[0] != expected_len:
            raise ValueError("batch priority score length mismatch")
        return np.maximum(arr, 0.0)

    @staticmethod
    def _subset_rows(
        state: MechanismRuntimeState,
        *,
        selected_positions: np.ndarray,
        batch_index_ref: np.ndarray,
        source_key: str,
    ) -> None:
        previous_n = int(state.X.shape[0])
        base_indices = np.arange(previous_n, dtype=int) if state.row_indices is None else np.asarray(state.row_indices, dtype=int)
        state.row_indices = base_indices[selected_positions]
        state.X = np.asarray(state.X[selected_positions], dtype=float)
        state.Y = np.asarray(state.Y[selected_positions], dtype=float)
        if state.sample_weight is not None:
            state.sample_weight = np.asarray(state.sample_weight[selected_positions], dtype=float)
        RowFeatureSubsampleMechanism._subset_signal_rows(state, selected_positions, previous_n)
        state.state_signals["sample_index_ref"] = np.asarray(state.row_indices, dtype=int)
        state.state_signals["batch_index_ref"] = np.asarray(batch_index_ref, dtype=int)
        state.metadata["batch_sampling"] = {
            "source_key": str(source_key),
            "selected_rows": int(state.X.shape[0]),
            "num_batches": int(np.max(batch_index_ref) + 1) if batch_index_ref.size > 0 else 0,
        }

    def pre_fit(self, state: MechanismRuntimeState) -> None:
        n = int(state.X.shape[0])
        if n <= 0:
            raise ValueError("batch priority sampling requires non-empty training data")

        source_key = str(self.params.get("source_key", "gradient_norm_ref"))
        raw = state.state_signals.get(source_key)
        if raw is None:
            for fallback_key in tuple(self.params.get("fallback_keys", ("per_sample_loss_ref", "sample_weight_ref", "uncertainty_ref"))):
                if str(fallback_key) == "sample_weight_ref" and state.sample_weight is not None:
                    raw = state.sample_weight
                    source_key = "sample_weight_ref"
                    break
                candidate = state.state_signals.get(str(fallback_key))
                if candidate is not None:
                    raw = candidate
                    source_key = str(fallback_key)
                    break
        if raw is None:
            state.record_trace(self.mechanism_key, kind=self.mechanism_kind, status="skipped", details={"reason": "missing_signal"})
            return

        score = self._flatten_scores(raw, expected_len=n)
        batch_size = max(1, int(self.params.get("batch_size", min(64, n))))
        num_batches = max(1, int(self.params.get("num_batches", 1)))
        target_count = min(n, max(1, int(self.params.get("target_count", batch_size * num_batches))))
        replace = bool(self.params.get("replace", False))
        mode = str(self.params.get("mode", "topk")).strip().lower() or "topk"
        rng = np.random.default_rng(self.params.get("random_seed"))

        if mode == "weighted":
            weights = score + float(self.params.get("floor", 1e-8))
            if not np.any(weights > 0.0):
                weights = np.ones_like(weights, dtype=float)
            probs = weights / np.sum(weights)
            selected_positions = np.asarray(rng.choice(n, size=target_count, replace=replace, p=probs), dtype=int)
        else:
            ranked = np.argsort(score)[::-1]
            if replace:
                top_k = ranked[: max(1, min(n, int(self.params.get("candidate_pool_size", min(n, max(target_count, batch_size * 2))))))]
                if top_k.size == 0:
                    top_k = np.arange(n, dtype=int)
                top_weights = score[top_k] + float(self.params.get("floor", 1e-8))
                top_probs = top_weights / np.sum(top_weights)
                selected_positions = np.asarray(rng.choice(top_k, size=target_count, replace=True, p=top_probs), dtype=int)
            else:
                selected_positions = np.asarray(ranked[:target_count], dtype=int)

        if bool(self.params.get("shuffle_selected", False)) and selected_positions.size > 1:
            rng.shuffle(selected_positions)

        if selected_positions.size > 1 and not replace and bool(self.params.get("preserve_unique", True)):
            selected_positions = np.asarray(list(dict.fromkeys(int(v) for v in selected_positions).keys()), dtype=int)
        if selected_positions.size == 0:
            state.record_trace(self.mechanism_key, kind=self.mechanism_kind, status="skipped", details={"reason": "empty_selection"})
            return

        effective_batches = max(1, int(np.ceil(float(selected_positions.size) / float(batch_size))))
        batch_index_ref = np.repeat(np.arange(effective_batches, dtype=int), batch_size)[: selected_positions.size]
        self._subset_rows(
            state,
            selected_positions=selected_positions,
            batch_index_ref=batch_index_ref,
            source_key=source_key,
        )
        selected_scores = score[np.asarray(np.clip(selected_positions, 0, n - 1), dtype=int)]
        state.record_trace(
            self.mechanism_key,
            kind=self.mechanism_kind,
            status="applied",
            details={
                "source_key": source_key,
                "mode": mode,
                "batch_size": int(batch_size),
                "num_batches": int(effective_batches),
                "selected_rows": int(selected_positions.size),
                "replace": bool(replace),
                "min_score": float(np.min(selected_scores)),
                "max_score": float(np.max(selected_scores)),
            },
        )


class RowFeatureSubsampleMechanism(RuntimeMechanismBase):
    mechanism_key = "sampling.row_feature_subsample"
    mechanism_kind = "sampling"

    @staticmethod
    def _resolve_count(total: int, *, fraction: Any, max_count: Any) -> int:
        count = int(total)
        if fraction is not None:
            frac = float(fraction)
            if frac <= 0.0:
                count = 1
            elif frac < 1.0:
                count = max(1, int(np.floor(float(total) * frac)))
            else:
                count = min(int(total), int(frac))
        if max_count is not None:
            count = min(count, max(1, int(max_count)))
        return max(1, min(int(total), int(count)))

    @staticmethod
    def _subset_signal_rows(state: MechanismRuntimeState, selected_positions: np.ndarray, previous_n: int) -> None:
        for key, value in list(state.state_signals.items()):
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == previous_n:
                state.state_signals[str(key)] = arr[selected_positions]

    def pre_fit(self, state: MechanismRuntimeState) -> None:
        rng = np.random.default_rng(self.params.get("random_seed"))
        trace: dict[str, Any] = {}

        previous_n = int(state.X.shape[0])
        row_count = self._resolve_count(
            previous_n,
            fraction=self.params.get("row_fraction"),
            max_count=self.params.get("max_rows"),
        )
        if row_count < previous_n:
            replace = bool(self.params.get("replace", False))
            selected_positions = np.sort(rng.choice(previous_n, size=row_count, replace=replace))
            base_indices = np.arange(previous_n, dtype=int) if state.row_indices is None else np.asarray(state.row_indices, dtype=int)
            state.row_indices = base_indices[selected_positions]
            state.X = np.asarray(state.X[selected_positions], dtype=float)
            state.Y = np.asarray(state.Y[selected_positions], dtype=float)
            if state.sample_weight is not None:
                state.sample_weight = np.asarray(state.sample_weight[selected_positions], dtype=float)
            self._subset_signal_rows(state, selected_positions, previous_n)
            trace.update({"selected_rows": int(row_count), "replace": bool(replace)})

        previous_d = int(state.X.shape[1])
        feature_count = self._resolve_count(
            previous_d,
            fraction=self.params.get("feature_fraction"),
            max_count=self.params.get("max_features"),
        )
        if feature_count < previous_d:
            feature_replace = bool(self.params.get("feature_replace", False))
            selected_feature_positions = np.sort(rng.choice(previous_d, size=feature_count, replace=feature_replace))
            base_feature_indices = (
                np.arange(previous_d, dtype=int) if state.feature_indices is None else np.asarray(state.feature_indices, dtype=int)
            )
            state.feature_indices = base_feature_indices[selected_feature_positions]
            state.X = np.asarray(state.X[:, selected_feature_positions], dtype=float)
            if state.feature_names:
                names = tuple(str(name) for name in state.feature_names)
                state.feature_names = tuple(names[int(i)] for i in selected_feature_positions)
            trace.update({"selected_features": int(feature_count), "feature_replace": bool(feature_replace)})

        if not trace:
            state.record_trace(self.mechanism_key, kind=self.mechanism_kind, status="skipped", details={"reason": "full_pass"})
            return
        state.record_trace(self.mechanism_key, kind=self.mechanism_kind, status="applied", details=trace)


class EnsembleAggregationSummaryMechanism(RuntimeMechanismBase):
    mechanism_key = "aggregation.ensemble_summary"
    mechanism_kind = "aggregation"

    @staticmethod
    def _num_boosted_rounds(model: Any) -> int | list[int] | None:
        if model is None:
            return None
        if hasattr(model, "get_booster"):
            return int(model.get_booster().num_boosted_rounds())
        estimators = tuple(getattr(model, "estimators", ()) or ())
        if estimators and all(hasattr(est, "get_booster") for est in estimators):
            return [int(est.get_booster().num_boosted_rounds()) for est in estimators]
        fitted_estimators = tuple(getattr(model, "estimators_", ()) or ())
        if fitted_estimators and all(hasattr(est, "get_booster") for est in fitted_estimators):
            return [int(est.get_booster().num_boosted_rounds()) for est in fitted_estimators]
        return None

    def post_fit(self, state: MechanismRuntimeState, *, model: Any, artifact_metadata: dict[str, Any]) -> None:
        estimator_count = int(len(tuple(getattr(model, "estimators_", ()) or ())))
        if estimator_count <= 0:
            estimator_count = int(len(tuple(getattr(model, "estimators", ()) or ())))
        summary = {
            "trainer_key": str(state.trainer_key),
            "family_key": str(state.family_key),
            "active_signal_keys": sorted(str(key) for key in state.state_signals.keys()),
            "selected_rows": None if state.row_indices is None else int(np.asarray(state.row_indices).shape[0]),
            "selected_features": None if state.feature_indices is None else int(np.asarray(state.feature_indices).shape[0]),
            "estimator_count": estimator_count,
        }
        if state.sample_weight is not None:
            weights = np.asarray(state.sample_weight, dtype=float).reshape(-1)
            summary["sample_weight_sum"] = float(np.sum(weights))
            summary["sample_weight_mean"] = float(np.mean(weights))
        estimator_weights = np.asarray(getattr(model, "estimator_weights_", ()), dtype=float).reshape(-1)
        if estimator_weights.size > 0:
            summary["estimator_weight_sum"] = float(np.sum(estimator_weights))
            summary["estimator_weight_count"] = int(estimator_weights.size)
        boosted_rounds = self._num_boosted_rounds(model)
        if boosted_rounds is not None:
            summary["num_boosted_rounds"] = boosted_rounds

        runtime_block = dict(artifact_metadata.get("runtime_mechanisms", {}))
        runtime_block["aggregation_summary"] = summary
        runtime_block["trace"] = list(state.metadata.get("mechanism_trace", []))
        artifact_metadata["runtime_mechanisms"] = runtime_block
        state.record_trace(self.mechanism_key, kind=self.mechanism_kind, status="applied", details=summary)
        artifact_metadata["runtime_mechanisms"]["trace"] = list(state.metadata.get("mechanism_trace", []))


_RUNTIME_MECHANISM_FACTORIES = {
    "state_signal_view.prediction_residual": PredictionResidualStateSignalView,
    "state_signal_view.prediction_loss": PredictionResidualStateSignalView,
    "state_signal_view.gradient_norm": GradientNormStateSignalView,
    "sample_weighting.loss_adaptive": LossAdaptiveSampleWeightingMechanism,
    "sample_weighting.loss_rank": LossAdaptiveSampleWeightingMechanism,
    "sampling.batch_priority_subsample": BatchPrioritySubsampleMechanism,
    "sampling.batch_subsample": BatchPrioritySubsampleMechanism,
    "sampling.row_feature_subsample": RowFeatureSubsampleMechanism,
    "sampling.subsample": RowFeatureSubsampleMechanism,
    "aggregation.ensemble_summary": EnsembleAggregationSummaryMechanism,
    "aggregation.summary": EnsembleAggregationSummaryMechanism,
}


def coerce_runtime_mechanism_spec(value: RuntimeMechanismSpec | Mapping[str, Any] | str) -> RuntimeMechanismSpec:
    if isinstance(value, RuntimeMechanismSpec):
        return value
    if isinstance(value, str):
        return RuntimeMechanismSpec(key=value, params={})
    raw = dict(value)
    return RuntimeMechanismSpec(
        key=str(raw.get("key", raw.get("mechanism_key", "noop"))),
        params=dict(raw.get("params", {})),
    )


def build_runtime_mechanism(value: RuntimeMechanismSpec | Mapping[str, Any] | str | RuntimeMechanismBase) -> RuntimeMechanismBase:
    if isinstance(value, RuntimeMechanismBase):
        return value
    spec = coerce_runtime_mechanism_spec(value)
    factory = _RUNTIME_MECHANISM_FACTORIES.get(str(spec.key))
    if factory is None:
        raise KeyError(f"unknown runtime mechanism key: {spec.key}")
    return factory(**dict(spec.params))


class MechanismRuntimeStack:
    def __init__(self, components: Sequence[RuntimeMechanismBase] = tuple()) -> None:
        ordered = sorted(
            tuple(components),
            key=lambda item: (MECHANISM_RUNTIME_ORDER.get(str(item.mechanism_kind), 99), str(item.mechanism_key)),
        )
        self.components = tuple(ordered)

    def summaries(self) -> list[dict[str, Any]]:
        return [component.summary() for component in self.components]

    def run_pre_fit(self, state: MechanismRuntimeState) -> None:
        for component in self.components:
            if str(component.mechanism_kind) == "aggregation":
                continue
            component.pre_fit(state)

    def run_post_fit(self, state: MechanismRuntimeState, *, model: Any, artifact_metadata: dict[str, Any]) -> None:
        for component in self.components:
            component.post_fit(state, model=model, artifact_metadata=artifact_metadata)


def build_runtime_mechanisms(
    values: Sequence[RuntimeMechanismSpec | Mapping[str, Any] | str | RuntimeMechanismBase] | None,
) -> MechanismRuntimeStack:
    if values is None:
        return MechanismRuntimeStack()
    return MechanismRuntimeStack(tuple(build_runtime_mechanism(value) for value in tuple(values)))


__all__ = [
    "BatchPrioritySubsampleMechanism",
    "EnsembleAggregationSummaryMechanism",
    "GradientNormStateSignalView",
    "LossAdaptiveSampleWeightingMechanism",
    "MechanismRuntimeStack",
    "MechanismRuntimeState",
    "PredictionResidualStateSignalView",
    "RowFeatureSubsampleMechanism",
    "RuntimeMechanismBase",
    "RuntimeMechanismSpec",
    "build_runtime_mechanism",
    "build_runtime_mechanisms",
    "coerce_runtime_mechanism_spec",
]
