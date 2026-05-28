"""Optional framework integration surfaces.

Import integration symbols from their submodules directly, e.g.::

    from mlblack.integrations.etf_temporal_forecast import EtfTemporalForecastConfig
    from mlblack.integrations.nsgablack_symbolic import OrthogonalBasisSetArtifact
"""

from .transformers_bridge import (
    PretrainedCheckpointMapper,
    PretrainedCheckpointMappingConfig,
    PretrainedCheckpointMappingReport,
    PretrainedModelBridge,
    PretrainedModelBridgeConfig,
    PretrainedTokenizerBridge,
    PretrainedTokenizerBridgeConfig,
)
from .nsgablack_neural import (
    TransformerSpecEvaluationRecord,
    TransformerSpecSearchConfig,
    TransformerSpecSearchProblem,
    TransformerSpecSearchSpace,
)

__all__ = [
    "PretrainedCheckpointMapper",
    "PretrainedCheckpointMappingConfig",
    "PretrainedCheckpointMappingReport",
    "PretrainedModelBridge",
    "PretrainedModelBridgeConfig",
    "PretrainedTokenizerBridge",
    "PretrainedTokenizerBridgeConfig",
    "TransformerSpecEvaluationRecord",
    "TransformerSpecSearchConfig",
    "TransformerSpecSearchProblem",
    "TransformerSpecSearchSpace",
]
