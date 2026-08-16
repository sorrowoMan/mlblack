from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import NumericDataView


class _AddConst:
    def __init__(self, delta: float) -> None:
        self.delta = float(delta)

    def transform(self, data, context=None):
        _ = context
        if isinstance(data, NumericDataView):
            return NumericDataView(
                X_train=data.X_train + self.delta,
                y_train=data.y_train,
                X_valid=None if data.X_valid is None else data.X_valid + self.delta,
                y_valid=data.y_valid,
                feature_names=data.feature_names,
                target_name=data.target_name,
                metadata=data.metadata,
            )
        return np.asarray(data, dtype=float) + self.delta


def _tiny_view() -> NumericDataView:
    return NumericDataView(
        X_train=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        y_train=np.array([0.0, 1.0], dtype=float),
    )


def test_ml_pipeline_slot_kernel_serial_transform() -> None:
    from mlblack.pipeline import build_pipeline_kernel

    kernel = build_pipeline_kernel(
        {
            "slots": (
                {
                    "slot": "transform",
                    "mode": "serial",
                    "operators": ("plus_one", "plus_two"),
                },
            )
        },
        operator_registry={
            "plus_one": _AddConst(1.0),
            "plus_two": _AddConst(2.0),
        },
    )

    out = kernel.data_pipeline.fit_transform(_tiny_view())
    assert np.allclose(out.X_train, np.array([[4.0, 5.0], [6.0, 7.0]], dtype=float))


def test_ml_pipeline_slot_kernel_router_transform() -> None:
    from mlblack.pipeline import build_pipeline_kernel

    kernel = build_pipeline_kernel(
        {
            "slots": (
                {
                    "slot": "transform",
                    "mode": "router",
                    "selector_key": "route",
                    "routes": {"a": "plus_one", "b": "plus_three"},
                    "strict": True,
                },
            )
        },
        operator_registry={
            "plus_one": _AddConst(1.0),
            "plus_three": _AddConst(3.0),
        },
    )

    out_a = kernel.run_slot("transform", _tiny_view(), {"route": "a"})
    out_b = kernel.run_slot("transform", _tiny_view(), {"route": "b"})
    assert np.allclose(out_a.X_train, np.array([[2.0, 3.0], [4.0, 5.0]], dtype=float))
    assert np.allclose(out_b.X_train, np.array([[4.0, 5.0], [6.0, 7.0]], dtype=float))


def test_ml_pipeline_slot_kernel_method_override_for_head() -> None:
    from mlblack.pipeline import build_pipeline_kernel

    class _Head:
        def predict(self, values, context=None):
            _ = context
            return np.asarray(values, dtype=float) + 9.0

    kernel = build_pipeline_kernel(
        {
            "slots": (
                {
                    "slot": "head",
                    "method": "predict",
                    "operators": ("head_main",),
                },
            )
        },
        operator_registry={"head_main": _Head()},
    )
    out = kernel.run_slot("head", np.array([1.0, 2.0]), {})
    assert np.allclose(out, [10.0, 11.0])
