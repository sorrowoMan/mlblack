from __future__ import annotations

import unittest

import numpy as np

from core.common.contracts import ProcessedDataset
from core.trainers.symbolic_stagewise_trainer import (
    SymbolicStagewiseSurrogateTrainer,
    SymbolicStagewiseTrainerConfig,
)

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "PyTorch is required for Adam/LBFGS inner optimization test")
class TestSymbolicStagewiseInnerOpt(unittest.TestCase):
    def test_fixed_structure_inner_opt_improves_or_matches_train_rmse(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(220, 4))
        w_true = np.asarray([[2.0], [-1.3], [0.6], [0.9]], dtype=float)
        y = X @ w_true + 0.35

        data = ProcessedDataset(
            X_train=np.asarray(X, dtype=float),
            y_train=np.asarray(y, dtype=float),
            feature_names=("x0", "x1", "x2", "x3"),
            target_names=("y",),
        )

        cfg_base = SymbolicStagewiseTrainerConfig(
            force_linear_base="on",
            search_max_added_terms=0,
            search_ridge_l2=8.0,
            keep_search_trace=False,
            search_graph_cache_enabled=False,
            search_path_memory_enabled=False,
            search_inner_opt_enabled=False,
        )
        trainer_base = SymbolicStagewiseSurrogateTrainer(config=cfg_base)
        art_base = trainer_base.fit(data)
        pred_base = art_base.predict(np.asarray(X, dtype=float))
        rmse_base = float(np.sqrt(np.mean((pred_base - y) ** 2)))

        cfg_inner = SymbolicStagewiseTrainerConfig(
            force_linear_base="on",
            search_max_added_terms=0,
            search_ridge_l2=8.0,
            keep_search_trace=False,
            search_graph_cache_enabled=False,
            search_path_memory_enabled=False,
            search_inner_opt_enabled=True,
            search_inner_opt_method="adam_lbfgs",
            search_inner_opt_adam_steps=180,
            search_inner_opt_adam_lr=2e-2,
            search_inner_opt_lbfgs_steps=60,
            search_inner_opt_lbfgs_lr=0.8,
            search_inner_opt_l2=0.0,
            search_inner_opt_accept_rmse_tol=1e-8,
        )
        trainer_inner = SymbolicStagewiseSurrogateTrainer(config=cfg_inner)
        art_inner = trainer_inner.fit(data)
        pred_inner = art_inner.predict(np.asarray(X, dtype=float))
        rmse_inner = float(np.sqrt(np.mean((pred_inner - y) ** 2)))

        self.assertLessEqual(rmse_inner, rmse_base + 1e-6)

        strategy = dict(art_inner.metadata.get("strategy", {}))
        inner_log = dict(strategy.get("inner_opt", {}))
        self.assertTrue(bool(inner_log.get("enabled_requested", False)))
        self.assertIn(str(inner_log.get("status", "")), {"applied", "rejected_rmse_guard", "failed"})
        if str(inner_log.get("status")) == "applied":
            self.assertTrue(bool(inner_log.get("applied", False)))


if __name__ == "__main__":
    unittest.main()

