from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import PreferencePairDataView


def build_data() -> PreferencePairDataView:
    return PreferencePairDataView(
        chosen_train=np.asarray(
            [[1, 2, 3, 4], [1, 3, 4, 5], [2, 3, 5, 6]],
            dtype=float,
        ),
        rejected_train=np.asarray(
            [[1, 2, 2, 2], [1, 3, 3, 3], [2, 3, 3, 3]],
            dtype=float,
        ),
    )


__all__ = ["build_data"]
