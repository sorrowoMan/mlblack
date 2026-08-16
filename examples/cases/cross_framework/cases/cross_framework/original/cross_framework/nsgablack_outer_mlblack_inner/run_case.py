from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlblack.pipeline.data_views import train_valid_split
from mlblack.problems import build_training_proxy


def simulated_outer_resource_context() -> dict:
    return {
        "scope": "nested-inner-training",
        "execution_backend": "serial",
        "compute_backend": "cpu",
        "device": "cpu",
        "threads": 1,
        "nested": True,
        "namespace": "nsgablack.outer.eval.0",
        "lease": {
            "lease_id": "simulated_outer_lease",
            "device_tokens": ["cpu:0"],
            "policy": "exclusive",
        },
        "metadata": {"source": "simulated_nsgablack_outer_allocator"},
    }


def main() -> None:
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = 0.25 + 1.75 * X[:, 0]
    data = train_valid_split(X, y, valid_ratio=0.25, feature_names=("x0",))
    proxy = build_training_proxy(
        data,
        trainer_spec={
            "preset": "orthogonal_linear_point",
            "run_name": "inner_mlblack_from_outer",
            "params": {"learning_rate": 0.05},
        },
        max_steps=8,
    )
    result = proxy.evaluate_individual([0.0], {"resource_context": simulated_outer_resource_context()})
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

