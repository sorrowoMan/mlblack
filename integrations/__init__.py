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
from .nsgablack_learning_case import (
    ComponentOverrideBuilder,
    NsgablackLearningCaseEvaluator,
    ResultProjector,
    project_learning_result,
)
from .nsgablack_solver_case import (
    BestSolutionProjector,
    DEFAULT_PROJECTABLE_SOLVE_STATUSES,
    NsgablackSolverCaseInvoker,
    OptimizationFeedbackMapper,
    ParetoFrontProjector,
    ParetoSelector,
    ParetoSolutionProjector,
    SolverCaseInvocationError,
    SolverCaseInvocationResult,
    SolverCaseProjection,
    SolverCaseResultProjector,
    SolverComponentOverrideBuilder,
    SolverFeedbackMapper,
)
from .nsgablack_symbolic_backend import (
    MlblackSymbolicConsensusBackend,
    MlblackSymbolicConsensusBackendConfig,
)
from .nsgablack_gradient import build_gradient_trainer
from .nsgablack_diagnostic import build_diagnostic_solver
from .nsgablack_control import (
    LearningSolver,
    MLLearningProblemBridge,
    MLRepresentationBridge,
    build_learning_solver,
)
from .nsgablack_optimization import build_optimization_adapter

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
    "ComponentOverrideBuilder",
    "NsgablackLearningCaseEvaluator",
    "ResultProjector",
    "project_learning_result",
    "BestSolutionProjector",
    "DEFAULT_PROJECTABLE_SOLVE_STATUSES",
    "NsgablackSolverCaseInvoker",
    "OptimizationFeedbackMapper",
    "ParetoFrontProjector",
    "ParetoSelector",
    "ParetoSolutionProjector",
    "SolverCaseInvocationError",
    "SolverCaseInvocationResult",
    "SolverCaseProjection",
    "SolverCaseResultProjector",
    "SolverComponentOverrideBuilder",
    "SolverFeedbackMapper",
    "MlblackSymbolicConsensusBackend",
    "MlblackSymbolicConsensusBackendConfig",
    "build_gradient_trainer",
    "build_diagnostic_solver",
    "LearningSolver",
    "MLLearningProblemBridge",
    "MLRepresentationBridge",
    "build_learning_solver",
    "build_optimization_adapter",
]
