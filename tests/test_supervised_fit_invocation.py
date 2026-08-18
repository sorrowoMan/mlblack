from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlblack.problems.supervised import _fit_estimator_with_lifecycle


def _model(*, strict: bool = False):
    return SimpleNamespace(
        mechanisms={
            "boosting": {
                "early_stopping": {
                    "rounds": 3,
                    "strict": strict,
                }
            }
        },
        params={},
    )


def _data():
    return SimpleNamespace(
        X_train=np.asarray([[1.0], [2.0]]),
        y_train=np.asarray([1.0, 2.0]),
        X_valid=np.asarray([[3.0]]),
        y_valid=np.asarray([3.0]),
    )


def test_estimator_fit_body_type_error_is_not_retried() -> None:
    class Estimator:
        def __init__(self) -> None:
            self.calls = 0

        def fit(self, X, y, **kwargs):
            del X, y, kwargs
            self.calls += 1
            raise TypeError("fit body failed")

    estimator = Estimator()
    with pytest.raises(TypeError, match="fit body failed"):
        _fit_estimator_with_lifecycle(estimator, _model(), _data(), {})

    assert estimator.calls == 1


def test_estimator_fit_uses_bound_legacy_form_once() -> None:
    class LegacyEstimator:
        def __init__(self) -> None:
            self.calls = 0

        def fit(self, X, y):
            del X, y
            self.calls += 1
            return self

    estimator = LegacyEstimator()
    audit = _fit_estimator_with_lifecycle(estimator, _model(), _data(), {})

    assert estimator.calls == 1
    assert audit["status"] == "fallback_no_fit_kwargs"
    assert audit["fit_kwargs"] == ()


def test_strict_fit_rejects_incompatible_signature_before_execution() -> None:
    class LegacyEstimator:
        def __init__(self) -> None:
            self.calls = 0

        def fit(self, X, y):
            del X, y
            self.calls += 1
            return self

    estimator = LegacyEstimator()
    with pytest.raises(TypeError, match="cannot bind"):
        _fit_estimator_with_lifecycle(
            estimator,
            _model(strict=True),
            _data(),
            {},
        )

    assert estimator.calls == 0
