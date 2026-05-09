from __future__ import annotations

import unittest

import numpy as np

from core.symbolic.symbolic_dsl import evaluate_expression_numpy, expression_to_string
from core.symbolic.symbolic_structure_search import (
    StructureSearchConfig,
    _build_nested_expr,
    _candidate_allowed,
    _expr_depth,
)


class TestSymbolicStructureNesting(unittest.TestCase):
    def test_legacy_nested_pattern_still_supported(self) -> None:
        base = {"type": "feature", "index": 0}
        expr = _build_nested_expr("sin(square)", base)
        self.assertIsNotNone(expr)
        txt = expression_to_string(expr, param_values=None, precision=8)
        self.assertEqual(txt.replace(" ", ""), "sin(((x0)^2))")

    def test_high_order_nested_pattern_supported(self) -> None:
        base = {"type": "feature", "index": 0}
        expr = _build_nested_expr("sin(square(tanh(x)))", base)
        self.assertIsNotNone(expr)
        self.assertEqual(_expr_depth(expr), 4)

        x = np.asarray([[0.1], [0.5], [1.0], [2.0]], dtype=float)
        y = evaluate_expression_numpy(expr, x)
        self.assertEqual(tuple(y.shape), (4,))
        self.assertTrue(bool(np.all(np.isfinite(y))))

    def test_compact_pattern_without_explicit_x_supported(self) -> None:
        base = {"type": "feature", "index": 0}
        expr = _build_nested_expr("exp(log(abs))", base)
        self.assertIsNotNone(expr)
        self.assertEqual(_expr_depth(expr), 4)

        x = np.asarray([[-2.0], [-0.5], [0.0], [1.2]], dtype=float)
        y = evaluate_expression_numpy(expr, x)
        self.assertTrue(bool(np.all(np.isfinite(y))))

    def test_invalid_pattern_returns_none(self) -> None:
        base = {"type": "feature", "index": 0}
        self.assertIsNone(_build_nested_expr("unknown(square)", base))
        self.assertIsNone(_build_nested_expr("sin(square(x)", base))

    def test_depth_gate_can_block_deep_nested_expr(self) -> None:
        base = {"type": "feature", "index": 0}
        expr = _build_nested_expr("sin(square(exp(log(abs(x)))))", base)
        self.assertIsNotNone(expr)
        self.assertGreaterEqual(_expr_depth(expr), 6)

        cfg_shallow = StructureSearchConfig(max_expr_depth=4)
        cfg_deep = StructureSearchConfig(max_expr_depth=8)
        self.assertFalse(_candidate_allowed(expr, cfg=cfg_shallow))
        self.assertTrue(_candidate_allowed(expr, cfg=cfg_deep))


if __name__ == "__main__":
    unittest.main()
