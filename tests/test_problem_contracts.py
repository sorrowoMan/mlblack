from __future__ import annotations

import numpy as np

from problem import BatchEvaluationProxyProvider, DecisionEvaluationBridge


def test_decision_evaluation_bridge_batch_path() -> None:
    def decode_fn(x: np.ndarray) -> tuple[list[int], int, dict[str, float]]:
        return [0, 1], 2, {"score": float(np.sum(x))}

    def evaluate_batch(decoded_batch):
        return np.asarray(
            [[float(len(item.subset_idx)), float(item.meta["score"]), 0.0] for item in decoded_batch],
            dtype=float,
        )

    bridge = DecisionEvaluationBridge(
        decode_fn=decode_fn,
        evaluate_decoded_fn=lambda item: np.asarray([9.0, 9.0, 9.0], dtype=float),
        evaluate_decoded_batch_fn=evaluate_batch,
        objective_dim=3,
    )

    obj, vio = bridge.evaluate_population(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    assert obj.shape == (2, 3)
    assert vio.shape == (2,)
    assert np.allclose(obj[:, 0], [2.0, 2.0])
    assert np.allclose(obj[:, 1], [3.0, 7.0])


def test_batch_evaluation_proxy_provider_forwards_population() -> None:
    provider = BatchEvaluationProxyProvider(
        evaluate_population_fn=lambda pop: (
            np.full((int(pop.shape[0]), 3), 1.5, dtype=float),
            np.zeros((int(pop.shape[0]),), dtype=float),
        )
    )
    pop = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    obj, vio = provider.evaluate_population(None, pop, {})
    assert obj.shape == (2, 3)
    assert vio.shape == (2,)
    assert np.allclose(obj, 1.5)
