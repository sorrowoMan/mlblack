from .assembly import BiasSpec, InnerTrainingAssemblySpec, TrainerAssemblySpec, build_pipeline, build_trainer

__version__ = "0.4.1"

from .bias import (
    BranchPolicyBias,
    DynamicPoolBias,
    L2ScaleBias,
    NoopBias,
    ObjectivePolicyBias,
    ObjectiveWeightBias,
    OptimizationBias,
    StateL2Bias,
)
from .core.artifacts import (
    ArtifactBuilder,
    ArtifactBundle,
    EstimatorStateArtifact,
    ModelArtifact,
    RunReport,
    SklearnMLPArtifact,
    TorchModelArtifact,
    TrainerStateArtifact,
    TreeEnsembleArtifact,
    TypedModelArtifact,
    XGBoostArtifact,
    load_artifact_bundle,
    save_artifact_bundle,
)
from blackbase.context import InMemoryContextStore, InMemorySnapshotStore
from blackbase.contracts import ComponentContract, ContractMixin
from blackbase.resources import PoolScheduler, ResourceContext
from .core.head import HeadBlock, OutputHead
from .core.state import TrainerState, build_trainer_state, replay_trainer, restore_trainer_state
from .core.types import Feedback, PopulationSnapshot, TrainerResult, UnknownState
from .integrations.nsgablack_control import LearningSolver, build_learning_solver
from .assembly.schema import DatasetSchema, FeatureSpec, ScaffoldConfig, TargetSpec
from .problems.training import TrainingContract, TrainingResultRecord, TrainingTask

__all__ = [
    "ArtifactBuilder",
    "ArtifactBundle",
    "BiasSpec",
    "BranchPolicyBias",
    "ComponentContract",
    "ContractMixin",
    "DatasetSchema",
    "DynamicPoolBias",
    "EstimatorStateArtifact",
    "FeatureSpec",
    "Feedback",
    "HeadBlock",
    "InMemoryContextStore",
    "InMemorySnapshotStore",
    "InnerTrainingAssemblySpec",
    "L2ScaleBias",
    "LearningSolver",
    "ModelArtifact",
    "NoopBias",
    "ObjectivePolicyBias",
    "ObjectiveWeightBias",
    "OptimizationBias",
    "OutputHead",
    "PopulationSnapshot",
    "ResourceContext",
    "RunReport",
    "ScaffoldConfig",
    "SklearnMLPArtifact",
    "StateL2Bias",
    "TargetSpec",
    "TorchModelArtifact",
    "TrainerAssemblySpec",
    "TrainerResult",
    "TrainerState",
    "TrainerStateArtifact",
    "TrainingContract",
    "TrainingResultRecord",
    "TrainingTask",
    "TreeEnsembleArtifact",
    "TypedModelArtifact",
    "UnknownState",
    "XGBoostArtifact",
    "__version__",
    "build_learning_solver",
    "build_pipeline",
    "build_trainer",
    "build_trainer_state",
    "load_artifact_bundle",
    "replay_trainer",
    "restore_trainer_state",
    "save_artifact_bundle",
]
