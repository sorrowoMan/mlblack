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
from core.common.trainer_shared import (
    PreparedTrainingData,
    prepare_training_data,
    resolve_feature_target_names,
    resolve_torch_device,
    set_torch_seed,
    split_train_val_indices,
)

__all__ = [
    "BaseSurrogateTrainer",
    "BatchStream",
    "BatchStreamSpec",
    "create_torch_batch_stream",
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
    "create_regression_objective",
    "create_quantile_objective",
    "create_torch_optimizer",
    "PreparedTrainingData",
    "prepare_training_data",
    "resolve_feature_target_names",
    "resolve_torch_device",
    "set_torch_seed",
    "split_train_val_indices",
]


def __getattr__(name: str):
    if name == "BaseSurrogateTrainer":
        from core.common.base_trainer import BaseSurrogateTrainer

        return BaseSurrogateTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
