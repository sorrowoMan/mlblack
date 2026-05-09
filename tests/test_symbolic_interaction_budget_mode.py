from __future__ import annotations

import unittest

import numpy as np

from core.symbolic.symbolic_structure_search import StructureSearchConfig, _build_candidates


class TestSymbolicInteractionBudgetMode(unittest.TestCase):
    def _toy_interaction_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(160, 4))
        r = (X[:, 0] * X[:, 1] + 0.05 * rng.normal(size=(160,))).reshape(-1, 1)
        return np.asarray(X, dtype=float), np.asarray(r, dtype=float)

    def test_interaction_first_reorders_candidates_under_budget(self) -> None:
        X, residual = self._toy_interaction_data()

        cfg_fixed = StructureSearchConfig(
            topk_features=2,
            unary_ops=("square",),
            nested_mode="manual",
            nested_unary_patterns=tuple(),
            include_hinge=False,
            enable_grad_residual_projection=False,
            max_pair_terms=1,
            max_candidates_per_iter=2,
            interaction_budget_mode="fixed",
            interaction_diag_threshold=0.5,
        )
        cands_fixed = _build_candidates(X, residual, cfg=cfg_fixed)
        fam_fixed = [str(c.get("family", "")) for c in cands_fixed]
        self.assertTrue(all(not f.startswith("interaction:") for f in fam_fixed))

        cfg_if = StructureSearchConfig(
            topk_features=2,
            unary_ops=("square",),
            nested_mode="manual",
            nested_unary_patterns=tuple(),
            include_hinge=False,
            enable_grad_residual_projection=False,
            max_pair_terms=1,
            max_candidates_per_iter=2,
            interaction_budget_mode="interaction_first",
            interaction_diag_threshold=0.5,
            interaction_pair_budget_boost=2.0,
        )
        cands_if = _build_candidates(X, residual, cfg=cfg_if)
        fam_if = [str(c.get("family", "")) for c in cands_if]
        self.assertTrue(any(f.startswith("interaction:") for f in fam_if))

    def test_budget_policy_metadata_present(self) -> None:
        X, residual = self._toy_interaction_data()
        cfg = StructureSearchConfig(
            topk_features=2,
            include_hinge=False,
            enable_grad_residual_projection=False,
            max_candidates_per_iter=8,
            interaction_budget_mode="interaction_first",
            interaction_diag_threshold=0.8,
        )
        cands = _build_candidates(X, residual, cfg=cfg)
        self.assertTrue(len(cands) > 0)
        meta = dict(cands[0].get("budget_policy", {}))
        self.assertEqual(str(meta.get("mode", "")), "interaction_first")
        self.assertIn("interaction_diag", meta)


if __name__ == "__main__":
    unittest.main()

