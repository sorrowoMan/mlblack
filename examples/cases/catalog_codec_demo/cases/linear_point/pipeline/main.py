"""生成公开合成数据；不读取业务数据或外部服务。"""
import numpy as np
from mlblack.pipeline.data_views import NumericDataView


def build_pipeline(*, seed=17, n_samples=24):
    rng = np.random.default_rng(int(seed))
    X = rng.uniform(-1.0, 1.0, size=(int(n_samples), 2))
    y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1]
    return NumericDataView(X_train=X, y_train=y, feature_names=("x0", "x1"))
