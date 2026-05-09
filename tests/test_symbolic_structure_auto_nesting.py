from __future__ import annotations

import unittest

import numpy as np

from core.symbolic.symbolic_structure_search import (
    StructureSearchConfig,
    _build_candidates,
    expression_to_string,
)


class TestSymbolicStructureAutoNesting(unittest.TestCase):
    def _toy_data(self) -> tuple[np.ndarray, np.ndarray]:
        x0 = np.linspace(-2.0, 2.0, 72)
        x1 = np.linspace(0.2, 3.0, 72)
        x2 = np.cos(np.linspace(0.0, 4.0, 72))
        X = np.asarray(np.stack([x0, x1, x2], axis=1), dtype=float)
        residual = (np.sin(x0) + 0.15 * x1).reshape(-1, 1)
        return X, residual

    def test_default_nested_mode_is_auto(self) -> None:
        cfg = StructureSearchConfig()
        self.assertEqual(str(cfg.nested_mode).strip().lower(), "auto")

    def test_auto_mode_generates_nested_auto_candidates(self) -> None:
        X, residual = self._toy_data()
        cfg = StructureSearchConfig(
            topk_features=1,
            max_pair_terms=0,
            include_hinge=False,
            enable_grad_residual_projection=False,
            nested_mode="auto",
            nested_unary_patterns=tuple(),
            auto_nested_allowed_ops=("square", "sin", "tanh"),
            auto_nested_min_depth=2,
            auto_nested_max_depth=3,
            auto_nested_beam_width=8,
            auto_nested_max_patterns_per_feature=12,
            max_candidates_per_iter=200,
            max_expr_depth=8,
        )
        cands = _build_candidates(X, residual, cfg=cfg)
        auto_nested = [c for c in cands if str(c.get("family", "")).startswith("nested:auto:")]
        self.assertTrue(len(auto_nested) > 0)
        max_depth = max(int(c.get("expr_depth", 0)) for c in auto_nested)
        self.assertGreaterEqual(max_depth, 3)

    def test_manual_mode_disables_auto_generation(self) -> None:
        X, residual = self._toy_data()
        cfg = StructureSearchConfig(
            topk_features=1,
            max_pair_terms=0,
            include_hinge=False,
            enable_grad_residual_projection=False,
            nested_mode="manual",
            nested_unary_patterns=tuple(),
            auto_nested_allowed_ops=("square", "sin", "tanh"),
            auto_nested_min_depth=2,
            auto_nested_max_depth=3,
            auto_nested_beam_width=8,
            auto_nested_max_patterns_per_feature=12,
            max_candidates_per_iter=200,
            max_expr_depth=8,
        )
        cands = _build_candidates(X, residual, cfg=cfg)
        auto_nested = [c for c in cands if str(c.get("family", "")).startswith("nested:auto:")]
        self.assertEqual(len(auto_nested), 0)

    def test_hybrid_mode_contains_manual_and_auto_nested(self) -> None:
        X, residual = self._toy_data()
        cfg = StructureSearchConfig(
            topk_features=1,
            max_pair_terms=0,
            include_hinge=False,
            enable_grad_residual_projection=False,
            nested_mode="hybrid",
            nested_unary_patterns=("sin(square)",),
            auto_nested_allowed_ops=("square", "sin", "tanh"),
            auto_nested_min_depth=2,
            auto_nested_max_depth=3,
            auto_nested_beam_width=8,
            auto_nested_max_patterns_per_feature=12,
            max_candidates_per_iter=200,
            max_expr_depth=8,
        )
        cands = _build_candidates(X, residual, cfg=cfg)
        auto_nested = [c for c in cands if str(c.get("family", "")).startswith("nested:auto:")]
        manual_nested = [c for c in cands if str(c.get("family", "")).startswith("nested:manual:")]
        self.assertTrue(len(auto_nested) > 0)
        self.assertTrue(len(manual_nested) > 0)

    def test_shared_full_candidate_pool_exposes_exp_and_ratio_families(self) -> None:
        x0 = np.linspace(0.5, 3.0, 96)
        x1 = np.linspace(1.2, 4.5, 96)
        x2 = np.cos(np.linspace(0.0, 3.0, 96))
        X = np.asarray(np.stack([x0, x1, x2], axis=1), dtype=float)
        residual = (1.4 * np.exp(-(x0 / x1)) + 0.1 * x2).reshape(-1, 1)
        cfg = StructureSearchConfig(
            candidate_pool_mode="shared_full",
            include_hinge=False,
            enable_grad_residual_projection=False,
            max_candidates_per_iter=400,
        )
        cands = _build_candidates(X, residual, cfg=cfg, feature_names=("ratio_core", "temperature", "aux"))
        exprs = [expression_to_string(dict(c["expr"]), precision=8) for c in cands]
        families = [str(c.get("family", "")) for c in cands]
        self.assertTrue(any("exp(" in expr for expr in exprs))
        self.assertTrue(any("ratio" in family or "rational" in family for family in families))
        self.assertTrue(all(str(c.get("budget_policy", {}).get("candidate_pool_mode", "")) == "shared_full" for c in cands))


if __name__ == "__main__":
    unittest.main()
