from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from core.common.contracts import ProcessedDataset


@dataclass(frozen=True)
class XGBoostTrainerConfig:
    artifact_id: str = "xgboost_baseline"
    n_estimators: int = 360
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    tree_method: str = "hist"
    random_seed: int = 42


class _SklearnFallbackRegressor:
    def __init__(self, *, random_seed: int = 42) -> None:
        self.random_seed = int(random_seed)
        self._model: Any | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnFallbackRegressor":
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.multioutput import MultiOutputRegressor
        except Exception:
            from sklearn.linear_model import Ridge
            self._model = Ridge(alpha=1.0)
            self._model.fit(X, y)
            return self

        yy = np.asarray(y, dtype=float)
        base = HistGradientBoostingRegressor(random_state=int(self.random_seed), max_iter=200)
        if yy.ndim == 2 and yy.shape[1] > 1:
            self._model = MultiOutputRegressor(base)
            self._model.fit(X, yy)
        else:
            self._model = base
            self._model.fit(X, yy.reshape(-1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return np.asarray(self._model.predict(X), dtype=float)


class XGBoostSurrogateArtifact:
    def __init__(
        self,
        *,
        artifact_id: str,
        model: Any,
        feature_names: Sequence[str] | None,
        target_names: Sequence[str] | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.artifact_id = str(artifact_id)
        self.model = model
        self.feature_names = tuple(feature_names or ())
        self.target_names = tuple(target_names or ())
        self.metadata = dict(metadata or {})

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.asarray(self.model.predict(np.asarray(X, dtype=float)), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


class XGBoostSurrogateTrainer:
    def __init__(self, config: XGBoostTrainerConfig | None = None) -> None:
        self.config = config or XGBoostTrainerConfig()

    def fit(self, data: ProcessedDataset) -> XGBoostSurrogateArtifact:
        X = np.asarray(data.X_train, dtype=float)
        y = np.asarray(data.y_train, dtype=float)
        try:
            from xgboost import XGBRegressor
            from sklearn.multioutput import MultiOutputRegressor

            model = XGBRegressor(
                n_estimators=int(self.config.n_estimators),
                max_depth=int(self.config.max_depth),
                learning_rate=float(self.config.learning_rate),
                subsample=float(self.config.subsample),
                colsample_bytree=float(self.config.colsample_bytree),
                tree_method=str(self.config.tree_method),
                random_state=int(self.config.random_seed),
                objective="reg:squarederror",
            )
            if y.ndim == 2 and y.shape[1] > 1:
                model = MultiOutputRegressor(model)
                model.fit(X, y)
            else:
                model.fit(X, y.reshape(-1))
        except Exception:
            model = _SklearnFallbackRegressor(random_seed=int(self.config.random_seed)).fit(X, y)

        return XGBoostSurrogateArtifact(
            artifact_id=str(self.config.artifact_id),
            model=model,
            feature_names=data.feature_names,
            target_names=data.target_names,
            metadata={"trainer": type(self).__name__},
        )

