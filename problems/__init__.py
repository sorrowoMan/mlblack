from .classification import SupervisedClassificationProblem
from .conditional import PiecewiseRegressionProblem
from .bridge import build_training_proxy, result_to_outer_tuple
from .proxy import MLBlackTrainingProxy
from .supervised import (
    SupervisedEstimatorFitRegressionProblem,
    SupervisedIntervalRegressionProblem,
    SupervisedRegressionProblem,
)
from .symbolic import FixedSymbolicRegressionProblem, OrthogonalBasisEvaluationProblem
from .time_series import RollingOriginForecastingProblem, TimeSeriesForecastingProblem
from .neural import (
    TemporalNeuralForecastingProblem,
    TemporalNeuralProbabilisticForecastingProblem,
    TemporalNeuralRollingOriginProblem,
    TinyCNNImageClassificationProblem,
    TinyCNNImageContrastiveProblem,
    TinyGNNGraphClassificationProblem,
    TinyTransformerClassificationProblem,
    TinyTransformerDPOPreferenceProblem,
    TinyTransformerLanguageModelProblem,
)
from .training import TrainingContract, TrainingLineage, TrainingResultRecord, TrainingTask

__all__ = [
    "MLBlackTrainingProxy",
    "PiecewiseRegressionProblem",
    "SupervisedEstimatorFitRegressionProblem",
    "FixedSymbolicRegressionProblem",
    "OrthogonalBasisEvaluationProblem",
    "SupervisedClassificationProblem",
    "SupervisedIntervalRegressionProblem",
    "SupervisedRegressionProblem",
    "RollingOriginForecastingProblem",
    "TimeSeriesForecastingProblem",
    "TemporalNeuralForecastingProblem",
    "TemporalNeuralProbabilisticForecastingProblem",
    "TemporalNeuralRollingOriginProblem",
    "TinyTransformerClassificationProblem",
    "TinyCNNImageClassificationProblem",
    "TinyCNNImageContrastiveProblem",
    "TinyGNNGraphClassificationProblem",
    "TinyTransformerDPOPreferenceProblem",
    "TinyTransformerLanguageModelProblem",
    "TrainingContract",
    "TrainingLineage",
    "TrainingResultRecord",
    "TrainingTask",
    "build_training_proxy",
    "result_to_outer_tuple",
]
