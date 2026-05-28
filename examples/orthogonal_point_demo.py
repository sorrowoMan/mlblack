from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mlblack.pipeline.data_views import train_valid_split
from mlblack.presets import build_orthogonal_linear_point_trainer
rng = np.random.default_rng(7)
X = rng.normal(size=(240, 2))
y = 2.0 + 3.0 * X[:, 0] + 4.0 * X[:, 1] + 1.5 * X[:, 0] * X[:, 1] + rng.normal(scale=0.05, size=240)
data = train_valid_split(X, y, valid_ratio=0.2, seed=42, feature_names=("x1", "x2"))
trainer = build_orthogonal_linear_point_trainer(
    data,
    learning_rate=0.05,
    l2=1e-4,
    max_components=5,
    run_name="demo_orthogonal_linear_point",
)
result = trainer.fit(max_steps=300)
print(result.report)


