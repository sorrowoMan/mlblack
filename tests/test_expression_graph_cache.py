from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.symbolic.symbolic_dsl import evaluate_expression_numpy
from core.symbolic.symbolic_gradient import evaluate_gradient_numpy


class TestExpressionGraphCache(unittest.TestCase):
    def test_value_cache_hits_after_first_eval(self) -> None:
        cache = ExpressionGraphCache(max_value_entries=16, max_derivative_entries=16)
        X = np.asarray([[0.1, 1.0], [0.4, 2.0], [0.8, 3.0]], dtype=float)
        expr = {
            "type": "binary",
            "op": "add",
            "left": {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 0}},
            "right": {"type": "binary", "op": "mul", "left": {"type": "feature", "index": 1}, "right": {"type": "feature", "index": 1}},
        }

        y0 = cache.evaluate_expression(expr, X, batch_key="fit")
        y1 = cache.evaluate_expression(expr, X, batch_key="fit")

        self.assertTrue(np.allclose(y0, y1))
        self.assertTrue(np.allclose(y0, evaluate_expression_numpy(expr, X)))
        st = cache.stats()
        self.assertGreaterEqual(int(st.value_hits), 1)
        self.assertGreaterEqual(int(st.value_misses), 1)

    def test_gradient_cache_reuses_derivative_expression(self) -> None:
        cache = ExpressionGraphCache(max_value_entries=32, max_derivative_entries=32)
        X = np.asarray([[0.2, 1.1], [0.3, 1.4], [0.9, 2.5], [1.2, 3.1]], dtype=float)
        expr = {
            "type": "binary",
            "op": "mul",
            "left": {"type": "feature", "index": 0},
            "right": {"type": "unary", "op": "tanh", "arg": {"type": "feature", "index": 1}},
        }

        g0 = evaluate_gradient_numpy(
            expr,
            X,
            feature_index=1,
            graph_cache=cache,
            batch_key="fit",
        )
        g1 = evaluate_gradient_numpy(
            expr,
            X,
            feature_index=1,
            graph_cache=cache,
            batch_key="fit",
        )

        self.assertTrue(np.allclose(g0, g1))
        g_ref = evaluate_gradient_numpy(expr, X, feature_index=1)
        self.assertTrue(np.allclose(g0, g_ref))
        st = cache.stats()
        self.assertGreaterEqual(int(st.derivative_hits), 1)
        self.assertGreaterEqual(int(st.derivative_misses), 1)
        self.assertGreaterEqual(int(st.value_hits), 1)

    def test_value_cache_lru_eviction(self) -> None:
        cache = ExpressionGraphCache(max_value_entries=1, max_derivative_entries=8)
        X = np.asarray([[1.0], [2.0], [3.0]], dtype=float)
        expr_a = {"type": "feature", "index": 0}
        expr_b = {"type": "unary", "op": "square", "arg": {"type": "feature", "index": 0}}

        cache.evaluate_expression(expr_a, X, batch_key="fit")
        cache.evaluate_expression(expr_b, X, batch_key="fit")
        cache.evaluate_expression(expr_a, X, batch_key="fit")

        st = cache.stats()
        # With max_value_entries=1, expr_a should be evicted after expr_b insert.
        self.assertGreaterEqual(int(st.value_misses), 3)
        self.assertLessEqual(int(st.value_entries), 1)

    def test_sqlite_derivative_persistence_across_runs(self) -> None:
        X = np.asarray([[0.2, 1.1], [0.3, 1.4], [0.9, 2.5], [1.2, 3.1]], dtype=float)
        expr = {
            "type": "binary",
            "op": "mul",
            "left": {"type": "feature", "index": 0},
            "right": {"type": "unary", "op": "tanh", "arg": {"type": "feature", "index": 1}},
        }

        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "expr_cache.sqlite3")

            cache1 = ExpressionGraphCache(
                backend="sqlite",
                db_path=db_path,
                namespace="unit_test",
                max_value_entries=8,
                max_derivative_entries=8,
            )
            g1 = evaluate_gradient_numpy(expr, X, feature_index=1, graph_cache=cache1, batch_key="fit")
            st1 = cache1.stats()
            cache1.close()

            self.assertGreaterEqual(int(st1.derivative_misses), 1)
            self.assertGreaterEqual(int(st1.persistent_derivative_writes), 1)

            cache2 = ExpressionGraphCache(
                backend="sqlite",
                db_path=db_path,
                namespace="unit_test",
                max_value_entries=8,
                max_derivative_entries=8,
            )
            g2 = evaluate_gradient_numpy(expr, X, feature_index=1, graph_cache=cache2, batch_key="fit")
            st2 = cache2.stats()
            cache2.close()

            self.assertTrue(np.allclose(g1, g2))
            self.assertGreaterEqual(int(st2.persistent_derivative_hits), 1)


if __name__ == "__main__":
    unittest.main()
