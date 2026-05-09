from core.artifacts.artifact import LinearSurrogateArtifact
from core.artifacts.artifact_persistence import ArtifactPersistenceBase
from core.common.base_trainer import BaseSurrogateTrainer
from core.common.batch_stream import BatchStream, BatchStreamSpec, create_torch_batch_stream
from core.common.contracts import Cell, ProcessedDataset, Sample, SampleDataset, SurrogateArtifact
from core.common.hypothesis_space import HypothesisSpace, TorchModuleHypothesisSpace
from core.common.loss_objective import (
    MSEObjective,
    PinballObjective,
    TrainingObjective,
    create_quantile_objective,
    create_regression_objective,
)
from core.common.param_optimizer import OptimizerSpec, create_torch_optimizer
from core.artifacts.tree_ensemble_artifact import TreeEnsembleSurrogateArtifact
from core.artifacts.sklearn_mlp_artifact import SklearnMLPSurrogateArtifact
from core.trainers.adaboost_trainer import AdaBoostSurrogateTrainer, AdaBoostTrainerConfig
from core.trainers.sklearn_mlp_trainer import SklearnMLPTrainerConfig, SklearnMLPSurrogateTrainer
from core.artifacts.symbolic_artifact import SymbolicSurrogateArtifact
from core.symbolic.gradient_correction import GradientCorrection, GradientCorrectionConfig
from core.symbolic.gradient_parser import GradientParser, GradientSignal
from core.symbolic.path_memory import PathPrior, SymbolicPathMemory, default_path_memory_db
from core.symbolic.structure_optimizer import StructureOptimizer, StructureScoreConfig
from core.symbolic.symbolic_gradient import (
    differentiate_expression_wrt_feature,
    differentiate_expression_wrt_param,
    evaluate_gradient_numpy,
    gradient_formula_strings,
)
from core.artifacts.symbolic_interval_artifact import SymbolicIntervalSurrogateArtifact
from core.artifacts.piecewise_symbolic_interval_artifact import PiecewiseSymbolicIntervalSurrogateArtifact
from core.trainers.symbolic_stagewise_trainer import SymbolicStagewiseSurrogateTrainer, SymbolicStagewiseTrainerConfig
from core.symbolic.symbolic_structure_search import (
    StructureSearchConfig,
    StructureSearchResult,
    evaluate_genome_with_ridge,
    regression_metrics,
    residual_guided_structure_search,
)
from core.trainers.symbolic_torch_interval_trainer import SymbolicTorchIntervalTrainer, SymbolicTorchIntervalTrainerConfig
from core.trainers.symbolic_orthogonal_interval_trainer import (
    SymbolicOrthogonalIntervalSurrogateTrainer,
    SymbolicOrthogonalIntervalTrainerConfig,
)
from core.trainers.symbolic_torch_trainer import SymbolicTorchSurrogateTrainer, SymbolicTorchTrainerConfig
from core.artifacts.torch_artifact import TorchMLPSurrogateArtifact
from core.trainers.bagging_trainer import BaggingSurrogateTrainer, BaggingTrainerConfig
from core.trainers.extra_trees_trainer import ExtraTreesSurrogateTrainer, ExtraTreesTrainerConfig
from core.trainers.torch_trainer import TorchMLPTrainerConfig, TorchMLPSurrogateTrainer
from core.common.trainer_shared import (
    PreparedTrainingData,
    prepare_training_data,
    resolve_feature_target_names,
    resolve_torch_device,
    set_torch_seed,
    split_train_val_indices,
)
from core.neural.trainer_family import (
    NeuralBackendSpec,
    NeuralBackboneSpec,
    NeuralBatchingSpec,
    NeuralOptimizationSpec,
    NeuralTaskHeadSpec,
    NeuralTrainerFamilySpec,
    build_sklearn_mlp_family_spec,
    build_torch_mlp_family_spec,
    coerce_neural_family_spec,
)
from core.tree.trainer_family import (
    TreeEnsembleSpec,
    TreeRegularizationSpec,
    TreeSamplingSpec,
    TreeSplitSpec,
    TreeTaskHeadSpec,
    TreeTrainerFamilySpec,
    build_adaboost_family_spec,
    build_bagging_family_spec,
    build_extra_trees_family_spec,
    build_random_forest_family_spec,
    coerce_tree_family_spec,
)
from core.trainers.random_forest_trainer import RandomForestSurrogateTrainer, RandomForestTrainerConfig
from core.trainers.tree_ensemble_trainer import SklearnTreeEnsembleSurrogateTrainer, TreeEnsembleTrainerConfig
from core.trainers.trainer import RidgeTrainerConfig, RidgeSurrogateTrainer
from core.artifacts.xgboost_artifact import XGBoostSurrogateArtifact
from core.trainers.xgboost_trainer import XGBoostSurrogateTrainer, XGBoostTrainerConfig

__all__ = [
    "Cell",
    "Sample",
    "SampleDataset",
    "ProcessedDataset",
    "SurrogateArtifact",
    "HypothesisSpace",
    "TorchModuleHypothesisSpace",
    "TrainingObjective",
    "MSEObjective",
    "PinballObjective",
    "OptimizerSpec",
    "BatchStream",
    "BatchStreamSpec",
    "create_regression_objective",
    "create_quantile_objective",
    "create_torch_optimizer",
    "create_torch_batch_stream",
    "PreparedTrainingData",
    "prepare_training_data",
    "resolve_feature_target_names",
    "resolve_torch_device",
    "set_torch_seed",
    "split_train_val_indices",
    "NeuralBackendSpec",
    "NeuralBackboneSpec",
    "NeuralBatchingSpec",
    "NeuralOptimizationSpec",
    "NeuralTaskHeadSpec",
    "NeuralTrainerFamilySpec",
    "build_sklearn_mlp_family_spec",
    "build_torch_mlp_family_spec",
    "coerce_neural_family_spec",
    "BaseSurrogateTrainer",
    "LinearSurrogateArtifact",
    "ArtifactPersistenceBase",
    "TorchMLPSurrogateArtifact",
    "SklearnMLPSurrogateArtifact",
    "XGBoostSurrogateArtifact",
    "TreeEnsembleSurrogateArtifact",
    "GradientSignal",
    "GradientParser",
    "GradientCorrectionConfig",
    "GradientCorrection",
    "PathPrior",
    "SymbolicPathMemory",
    "default_path_memory_db",
    "StructureScoreConfig",
    "StructureOptimizer",
    "SymbolicSurrogateArtifact",
    "SymbolicIntervalSurrogateArtifact",
    "PiecewiseSymbolicIntervalSurrogateArtifact",
    "RidgeTrainerConfig",
    "RidgeSurrogateTrainer",
    "AdaBoostTrainerConfig",
    "AdaBoostSurrogateTrainer",
    "BaggingTrainerConfig",
    "BaggingSurrogateTrainer",
    "ExtraTreesTrainerConfig",
    "ExtraTreesSurrogateTrainer",
    "RandomForestTrainerConfig",
    "RandomForestSurrogateTrainer",
    "TreeEnsembleTrainerConfig",
    "SklearnTreeEnsembleSurrogateTrainer",
    "TorchMLPTrainerConfig",
    "TorchMLPSurrogateTrainer",
    "SklearnMLPTrainerConfig",
    "SklearnMLPSurrogateTrainer",
    "XGBoostTrainerConfig",
    "XGBoostSurrogateTrainer",
    "SymbolicTorchTrainerConfig",
    "SymbolicTorchSurrogateTrainer",
    "SymbolicTorchIntervalTrainerConfig",
    "SymbolicTorchIntervalTrainer",
    "SymbolicOrthogonalIntervalTrainerConfig",
    "SymbolicOrthogonalIntervalSurrogateTrainer",
    "SymbolicStagewiseTrainerConfig",
    "SymbolicStagewiseSurrogateTrainer",
    "differentiate_expression_wrt_param",
    "differentiate_expression_wrt_feature",
    "gradient_formula_strings",
    "evaluate_gradient_numpy",
    "TreeEnsembleSpec",
    "TreeSamplingSpec",
    "TreeSplitSpec",
    "TreeRegularizationSpec",
    "TreeTaskHeadSpec",
    "TreeTrainerFamilySpec",
    "build_adaboost_family_spec",
    "build_bagging_family_spec",
    "build_extra_trees_family_spec",
    "build_random_forest_family_spec",
    "coerce_tree_family_spec",
    "StructureSearchConfig",
    "StructureSearchResult",
    "residual_guided_structure_search",
    "evaluate_genome_with_ridge",
    "regression_metrics",
]

