from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class NumericizationPlan:
    """Frozen plan that describes how object cells map to numeric tensors."""

    feature_keys: tuple[str, ...]
    feature_sizes: Mapping[str, int]
    feature_names: tuple[str, ...]
    feature_modalities: Mapping[str, str]
    feature_states: Mapping[str, Mapping[str, Any]]
    target_key: str
    target_names: tuple[str, ...]
    target_codec_key: str
    target_codec_state: Mapping[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        feature_states: dict[str, dict[str, Any]] = {}
        for key, state in dict(self.feature_states).items():
            raw = dict(state)
            # runtime caches are not part of persisted metadata
            raw.pop("index_map", None)
            feature_states[str(key)] = raw

        return {
            "feature_keys": tuple(self.feature_keys),
            "feature_sizes": {str(k): int(v) for k, v in dict(self.feature_sizes).items()},
            "feature_names": tuple(self.feature_names),
            "feature_modalities": {str(k): str(v) for k, v in dict(self.feature_modalities).items()},
            "feature_states": feature_states,
            "target_key": str(self.target_key),
            "target_names": tuple(self.target_names),
            "target_codec": str(self.target_codec_key),
            "target_codec_state": dict(self.target_codec_state),
        }
