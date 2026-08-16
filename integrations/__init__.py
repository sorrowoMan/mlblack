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
from .nsgablack_pipeline_kernel import (
    MLPipelineKernelBuild,
    PipelineSlotSpec,
    PipelineSpec,
    build_pipeline_kernel,
)
from .nsgablack_trainer_evaluator import (
    ComponentOverrideBuilder,
    NsgablackTrainerCaseEvaluator,
    NsgablackTrainerEvaluator,
    ResultProjector,
    TrainerFactory,
    project_trainer_result,
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
    "MLPipelineKernelBuild",
    "PipelineSlotSpec",
    "PipelineSpec",
    "build_pipeline_kernel",
    "ComponentOverrideBuilder",
    "NsgablackTrainerCaseEvaluator",
    "NsgablackTrainerEvaluator",
    "ResultProjector",
    "TrainerFactory",
    "project_trainer_result",
]
