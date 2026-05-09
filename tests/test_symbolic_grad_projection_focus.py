from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.symbolic.symbolic_structure_search import (
    StructureSearchConfig,
    _build_grad_projection_candidates,
)


class TestSymbolicGradProjectionFocusTransform(unittest.TestCase):
    def _dummy_signal(self, X: np.ndarray) -> SimpleNamespace:
        x = np.asarray(X, dtype=float)
        n, d = x.shape
        u0 = (np.sin(x[:, 0]) + 0.05 * x[:, 1]).reshape(n, 1)
        zeros = np.zeros((n, 1), dtype=float)
        gaps = tuple(u0 if j == 0 else zeros for j in range(d))
        return SimpleNamespace(
            feature_priority=np.asarray([1.0, 0.6, 0.2], dtype=float),
            gap_by_feature=gaps,
        )

    def test_focus_transform_enabled_generates_transformed_focus_candidates(self) -> None:
        x0 = np.linspace(-2.5, 2.5, 48)
        x1 = np.linspace(0.5, 3.5, 48)
        x2 = np.cos(np.linspace(0.0, 3.0, 48))
        X = np.asarray(np.stack([x0, x1, x2], axis=1), dtype=float)
        is_binary = np.asarray([False, False, False], dtype=bool)

        cfg = StructureSearchConfig(
            enable_grad_residual_projection=True,
            grad_projection_focus_include_transforms=True,
            grad_projection_focus_topk_transforms=3,
            grad_projection_topk_focus=1,
            grad_projection_partner_pool=3,
            grad_projection_topk_partners=2,
            grad_projection_topk_unary=2,
            grad_projection_partner_orders=(1,),
            grad_projection_min_abs_corr=0.0,
            grad_projection_max_generated=80,
            max_arity=3,
            max_expr_depth=8,
        )
        signal = self._dummy_signal(X)
        out = _build_grad_projection_candidates(
            X=X,
            cfg=cfg,
            gradient_signal=signal,
            residual_selected=(0, 1, 2),
            is_binary=is_binary,
        )
        self.assertTrue(len(out) > 0)
        transformed = [c for c in out if bool(c.get("grad_projection", {}).get("focus_transformed", False))]
        self.assertTrue(len(transformed) > 0)
        self.assertTrue(
            all(str(c.get("family", "")).startswith("interaction:grad_projected_focus_order") for c in transformed)
        )

    def test_focus_transform_disabled_keeps_legacy_focus_path(self) -> None:
        x0 = np.linspace(-2.0, 2.0, 40)
        x1 = np.linspace(0.2, 2.8, 40)
        x2 = np.sin(np.linspace(0.0, 2.0, 40))
        X = np.asarray(np.stack([x0, x1, x2], axis=1), dtype=float)
        is_binary = np.asarray([False, False, False], dtype=bool)

        cfg = StructureSearchConfig(
            enable_grad_residual_projection=True,
            grad_projection_focus_include_transforms=False,
            grad_projection_focus_topk_transforms=3,
            grad_projection_topk_focus=1,
            grad_projection_partner_pool=3,
            grad_projection_topk_partners=2,
            grad_projection_topk_unary=2,
            grad_projection_partner_orders=(1,),
            grad_projection_min_abs_corr=0.0,
            grad_projection_max_generated=60,
            max_arity=3,
            max_expr_depth=8,
        )
        signal = self._dummy_signal(X)
        out = _build_grad_projection_candidates(
            X=X,
            cfg=cfg,
            gradient_signal=signal,
            residual_selected=(0, 1, 2),
            is_binary=is_binary,
        )
        self.assertTrue(len(out) > 0)
        self.assertTrue(all(not bool(c.get("grad_projection", {}).get("focus_transformed", False)) for c in out))


if __name__ == "__main__":
    unittest.main()

