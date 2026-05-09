# CURRENT_STATE

- Generated at: 2026-04-14T22:25:37
- Root: `C:/Users/hp/Desktop/mlblack`
- Scan type: full source API scan (AST)
- Included roots: `bias/`, `config/`, `core/`, `examples/`, `numericizer/`, `pipeline/`, `project/`, `schema/`, `workflow/`
- Excluded: `__pycache__/`, `examples/out/` generated artifacts
- Parsed modules: **81**

## 1) Capability Overview

- Training backends: `ridge`, `torch_mlp`, `sklearn_mlp`, `xgboost`, `symbolic_torch`, `symbolic_torch_interval`, `symbolic_stagewise`
- Symbolic stack: DSL, symbolic gradients, gradient parser/correction, structure optimizer/search, path memory
- Orchestration/config: registry + assembly spec + flow assembly + workflow entry
- Data protocols: `ProcessedDataset` / `SampleDataset` / typed schema + numericizer + pipeline
- Example runners: benchmark/bridge/report/evaluation scripts under `examples/`

## 2) Package Entry Exports (`__init__`)

### `config.__init__`
- `__all__`: `ComponentRegistry`, `MLBlackConfig`, `create_default_config`, `SEMANTIC_NUMERICIZER_KEYS`, `BiasSpec`, `NumericizerSpec`, `TrainerAssemblySpec`, `FlowAssemblySpec`, `validate_flow_assembly`, `build_numericizer`, `build_trainer`, `build_flow_components`, `list_registered`, `describe_registered`, `describe_trainers`

### `core.__init__`
- `__all__`: `Cell`, `Sample`, `SampleDataset`, `ProcessedDataset`, `SurrogateArtifact`, `HypothesisSpace`, `TorchModuleHypothesisSpace`, `TrainingObjective`, `MSEObjective`, `PinballObjective`, `OptimizerSpec`, `BatchStream`, `BatchStreamSpec`, `create_regression_objective`, `create_quantile_objective`, `create_torch_optimizer`, `create_torch_batch_stream`, `PreparedTrainingData`, `prepare_training_data`, `resolve_feature_target_names`, `resolve_torch_device`, `set_torch_seed`, `split_train_val_indices`, `BaseSurrogateTrainer`, `LinearSurrogateArtifact`, `ArtifactPersistenceBase`, `TorchMLPSurrogateArtifact`, `SklearnMLPSurrogateArtifact`, `XGBoostSurrogateArtifact`, `GradientSignal`, `GradientParser`, `GradientCorrectionConfig`, `GradientCorrection`, `PathPrior`, `SymbolicPathMemory`, `default_path_memory_db`, `StructureScoreConfig`, `StructureOptimizer`, `SymbolicSurrogateArtifact`, `SymbolicIntervalSurrogateArtifact`, `RidgeTrainerConfig`, `RidgeSurrogateTrainer`, `TorchMLPTrainerConfig`, `TorchMLPSurrogateTrainer`, `SklearnMLPTrainerConfig`, `SklearnMLPSurrogateTrainer`, `XGBoostTrainerConfig`, `XGBoostSurrogateTrainer`, `SymbolicTorchTrainerConfig`, `SymbolicTorchSurrogateTrainer`, `SymbolicTorchIntervalTrainerConfig`, `SymbolicTorchIntervalTrainer`, `SymbolicStagewiseTrainerConfig`, `SymbolicStagewiseSurrogateTrainer`, `differentiate_expression_wrt_param`, `differentiate_expression_wrt_feature`, `gradient_formula_strings`, `evaluate_gradient_numpy`, `StructureSearchConfig`, `StructureSearchResult`, `residual_guided_structure_search`, `evaluate_genome_with_ridge`, `regression_metrics`

### `core.trainers.__init__`
- `__all__`: `RidgeTrainerConfig`, `RidgeSurrogateTrainer`, `TorchMLPTrainerConfig`, `TorchMLPSurrogateTrainer`, `SklearnMLPTrainerConfig`, `SklearnMLPSurrogateTrainer`, `XGBoostTrainerConfig`, `XGBoostSurrogateTrainer`, `SymbolicTorchTrainerConfig`, `SymbolicTorchSurrogateTrainer`, `SymbolicTorchIntervalTrainerConfig`, `SymbolicTorchIntervalTrainer`, `SymbolicStagewiseTrainerConfig`, `SymbolicStagewiseSurrogateTrainer`

### `core.symbolic.__init__`
- `__all__`: `GradientSignal`, `GradientParser`, `GradientCorrectionConfig`, `GradientCorrection`, `PathPrior`, `SymbolicPathMemory`, `default_path_memory_db`, `StructureScoreConfig`, `StructureOptimizer`, `differentiate_expression_wrt_param`, `differentiate_expression_wrt_feature`, `gradient_formula_strings`, `evaluate_gradient_numpy`, `StructureSearchConfig`, `StructureSearchResult`, `residual_guided_structure_search`, `evaluate_genome_with_ridge`, `regression_metrics`

### `core.artifacts.__init__`
- `__all__`: `LinearSurrogateArtifact`, `ArtifactPersistenceBase`, `TorchMLPSurrogateArtifact`, `SklearnMLPSurrogateArtifact`, `XGBoostSurrogateArtifact`, `SymbolicSurrogateArtifact`, `SymbolicIntervalSurrogateArtifact`

### `numericizer.__init__`
- `__all__`: `BaseNumericizer`, `DefaultNumericizer`, `ModalityEncoder`, `NumericizationPlan`, `BaseTargetCodec`, `TargetCodec`, `TargetCodecError`, `NumericTargetCodec`, `BinaryTargetCodec`, `CategoricalTargetCodec`

### `pipeline.__init__`
- `__all__`: `BasePipeline`, `IdentityPipeline`, `ZScorePipeline`, `create_pipeline`

### `bias.__init__`
- `__all__`: `BaseTrainingBias`, `FitContext`, `NoOpBias`, `L2ScaleBias`

### `schema.__init__`
- `__all__`: `DatasetSchema`, `FeatureSpec`, `TargetSpec`, `SchemaValidationError`, `parse_row`, `parse_rows`, `ViewBuildError`, `build_target_view`, `build_target_views`

### `workflow.__init__`
- `__all__`: `BaseDataReader`, `MemoryDataReader`, `TrainDataBundle`, `TrainFlowSpec`, `SemanticTrainFlowSpec`, `TrainFlowResult`, `run_train_flow`, `run_semantic_train_flow`

### `project.__init__`
- `__all__`: `TableDataSpec`, `TrainStageSpec`, `ScaffoldSpec`, `build_scaffold_spec`, `init_project`, `load_scaffold_spec`, `run_project_scaffold`

## 3) Full Module API Index

Note: each module lists top-level classes/functions; classes include fields and method signatures.

### Package `bias`

#### Module `bias.__init__`
- File: `bias\__init__.py`
- `__all__`: `BaseTrainingBias`, `FitContext`, `NoOpBias`, `L2ScaleBias`
- Top-level classes: 0
- Top-level functions: 0

#### Module `bias.base`
- File: `bias\base.py`
- Top-level classes: 2
  - `class FitContext`
    - fields:
      - `l2_multiplier: float = 1.0`
      - `sample_weight: np.ndarray | None = None`
      - `metadata: Dict[str, Any] = field(default_factory=dict)`
  - `class BaseTrainingBias(ABC)`
    - methods:
      - `apply(self, X: np.ndarray, Y: np.ndarray, context: FitContext) -> tuple[np.ndarray, np.ndarray]`
- Top-level functions: 0

#### Module `bias.l2_scale`
- File: `bias\l2_scale.py`
- Top-level classes: 1
  - `class L2ScaleBias(BaseTrainingBias)`
    - methods:
      - `__init__(self, scale: float = 1.0) -> None`
      - `apply(self, X: np.ndarray, Y: np.ndarray, context: FitContext) -> tuple[np.ndarray, np.ndarray]`
- Top-level functions: 0

#### Module `bias.noop`
- File: `bias\noop.py`
- Top-level classes: 1
  - `class NoOpBias(BaseTrainingBias)`
    - methods:
      - `apply(self, X: np.ndarray, Y: np.ndarray, context: FitContext) -> tuple[np.ndarray, np.ndarray]`
- Top-level functions: 0

### Package `config`

#### Module `config.__init__`
- File: `config\__init__.py`
- `__all__`: `ComponentRegistry`, `MLBlackConfig`, `create_default_config`, `SEMANTIC_NUMERICIZER_KEYS`, `BiasSpec`, `NumericizerSpec`, `TrainerAssemblySpec`, `FlowAssemblySpec`, `validate_flow_assembly`, `build_numericizer`, `build_trainer`, `build_flow_components`, `list_registered`, `describe_registered`, `describe_trainers`
- Top-level classes: 0
- Top-level functions: 0

#### Module `config.assembly`
- File: `config\assembly.py`
- Top-level classes: 4
  - `class BiasSpec`
    - fields:
      - `key: str = 'noop'`
      - `params: Dict[str, Any] = field(default_factory=dict)`
  - `class NumericizerSpec`
    - fields:
      - `key: str = 'default'`
      - `params: Dict[str, Any] = field(default_factory=dict)`
  - `class TrainerAssemblySpec`
    - fields:
      - `trainer_key: str = 'ridge'`
      - `trainer_params: Dict[str, Any] = field(default_factory=dict)`
      - `pipeline_key: str = 'identity'`
      - `pipeline_params: Dict[str, Any] = field(default_factory=dict)`
      - `biases: Sequence[BiasSpec] = field(default_factory=tuple)`
  - `class FlowAssemblySpec`
    - fields:
      - `trainer: TrainerAssemblySpec = field(default_factory=TrainerAssemblySpec)`
      - `numericizer: NumericizerSpec | None = field(default_factory=NumericizerSpec)`
- Top-level functions: 7
  - `validate_flow_assembly(spec: FlowAssemblySpec) -> None`
  - `build_numericizer(spec: NumericizerSpec | None = None, config: MLBlackConfig | None = None)`
  - `build_trainer(spec: TrainerAssemblySpec, config: MLBlackConfig | None = None)`
  - `build_flow_components(spec: FlowAssemblySpec, config: MLBlackConfig | None = None) -> Dict[str, Any]`
  - `list_registered(config: MLBlackConfig | None = None) -> Dict[str, Sequence[str]]`
  - `describe_registered(config: MLBlackConfig | None = None) -> Dict[str, Sequence[Dict[str, Any]]]`
  - `describe_trainers(config: MLBlackConfig | None = None, *, include_dynamic: bool = True) -> Dict[str, Dict[str, Any]]`

#### Module `config.defaults`
- File: `config\defaults.py`
- Top-level classes: 0
- Top-level functions: 9
  - `_split_numericizer_options(cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Any, Any, Any, Any, Any]`
  - `_build_ridge_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | RidgeTrainerConfig | None = None)`
  - `_build_torch_mlp_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | TorchMLPTrainerConfig | None = None)`
  - `_build_sklearn_mlp_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | SklearnMLPTrainerConfig | None = None)`
  - `_build_symbolic_torch_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | SymbolicTorchTrainerConfig | None = None)`
  - `_build_symbolic_stagewise_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | SymbolicStagewiseTrainerConfig | None = None)`
  - `_build_symbolic_torch_interval_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | SymbolicTorchIntervalTrainerConfig | None = None)`
  - `_build_xgboost_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | object | None = None)`
  - `create_default_config() -> MLBlackConfig`

#### Module `config.registry`
- File: `config\registry.py`
- Top-level classes: 2
  - `class ComponentRegistry`
    - fields:
      - `kind: str`
      - `_factories: Dict[str, Factory] = field(default_factory=dict)`
      - `_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)`
    - methods:
      - `register(self, key: str, factory: Factory, *, replace: bool = False, metadata: Mapping[str, Any] | None = None) -> None`
      - `create(self, key: str, **kwargs: Any) -> Any`
      - `get(self, key: str) -> Factory | None`
      - `keys(self) -> Tuple[str, ...]`
      - `metadata(self, key: str) -> Dict[str, Any]`
      - `describe(self) -> Tuple[Dict[str, Any], ...]`
  - `class MLBlackConfig`
    - fields:
      - `pipelines: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind='pipeline'))`
      - `biases: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind='bias'))`
      - `numericizers: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind='numericizer'))`
      - `trainers: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind='trainer'))`
- Top-level functions: 1
  - `_normalize_key(key: str) -> str`

### Package `core`

#### Module `core.__init__`
- File: `core\__init__.py`
- `__all__`: `Cell`, `Sample`, `SampleDataset`, `ProcessedDataset`, `SurrogateArtifact`, `HypothesisSpace`, `TorchModuleHypothesisSpace`, `TrainingObjective`, `MSEObjective`, `PinballObjective`, `OptimizerSpec`, `BatchStream`, `BatchStreamSpec`, `create_regression_objective`, `create_quantile_objective`, `create_torch_optimizer`, `create_torch_batch_stream`, `PreparedTrainingData`, `prepare_training_data`, `resolve_feature_target_names`, `resolve_torch_device`, `set_torch_seed`, `split_train_val_indices`, `BaseSurrogateTrainer`, `LinearSurrogateArtifact`, `ArtifactPersistenceBase`, `TorchMLPSurrogateArtifact`, `SklearnMLPSurrogateArtifact`, `XGBoostSurrogateArtifact`, `GradientSignal`, `GradientParser`, `GradientCorrectionConfig`, `GradientCorrection`, `PathPrior`, `SymbolicPathMemory`, `default_path_memory_db`, `StructureScoreConfig`, `StructureOptimizer`, `SymbolicSurrogateArtifact`, `SymbolicIntervalSurrogateArtifact`, `RidgeTrainerConfig`, `RidgeSurrogateTrainer`, `TorchMLPTrainerConfig`, `TorchMLPSurrogateTrainer`, `SklearnMLPTrainerConfig`, `SklearnMLPSurrogateTrainer`, `XGBoostTrainerConfig`, `XGBoostSurrogateTrainer`, `SymbolicTorchTrainerConfig`, `SymbolicTorchSurrogateTrainer`, `SymbolicTorchIntervalTrainerConfig`, `SymbolicTorchIntervalTrainer`, `SymbolicStagewiseTrainerConfig`, `SymbolicStagewiseSurrogateTrainer`, `differentiate_expression_wrt_param`, `differentiate_expression_wrt_feature`, `gradient_formula_strings`, `evaluate_gradient_numpy`, `StructureSearchConfig`, `StructureSearchResult`, `residual_guided_structure_search`, `evaluate_genome_with_ridge`, `regression_metrics`
- Top-level classes: 0
- Top-level functions: 0

#### Module `core.artifacts.__init__`
- File: `core\artifacts\__init__.py`
- `__all__`: `LinearSurrogateArtifact`, `ArtifactPersistenceBase`, `TorchMLPSurrogateArtifact`, `SklearnMLPSurrogateArtifact`, `XGBoostSurrogateArtifact`, `SymbolicSurrogateArtifact`, `SymbolicIntervalSurrogateArtifact`
- Top-level classes: 0
- Top-level functions: 0

#### Module `core.artifacts.artifact`
- File: `core\artifacts\artifact.py`
- Top-level classes: 1
  - `class LinearSurrogateArtifact(ArtifactPersistenceBase)`
    - fields:
      - `artifact_id: str`
      - `coef: np.ndarray`
      - `intercept: np.ndarray`
      - `x_mean: np.ndarray`
      - `x_std: np.ndarray`
      - `residual_std: np.ndarray`
      - `feature_names: Sequence[str]`
      - `target_names: Sequence[str]`
      - `pipeline_name: str = 'identity'`
      - `pipeline_state: Dict[str, Any] = field(default_factory=dict)`
      - `ood_z_threshold: float = 4.0`
      - `metadata: Dict[str, Any] = field(default_factory=dict)`
    - methods:
      - `_transform(self, X: np.ndarray) -> np.ndarray`
      - `predict(self, X: np.ndarray) -> np.ndarray`
      - `uncertainty(self, X: np.ndarray) -> np.ndarray`
      - `validity(self, X: np.ndarray) -> np.ndarray`
      - `save(self, out_dir: str) -> None`
      - `load(cls, out_dir: str) -> 'LinearSurrogateArtifact'`
- Top-level functions: 1
  - `_as_2d(arr: np.ndarray) -> np.ndarray`

#### Module `core.artifacts.artifact_persistence`
- File: `core\artifacts\artifact_persistence.py`
- Top-level classes: 1
  - `class ArtifactPersistenceBase`
    - methods:
      - `_ensure_dir(out_dir: str) -> Path`
      - `_save_npz(path: Path, filename: str, **arrays: Any) -> None`
      - `_load_npz(path: Path, filename: str)`
      - `_save_json(path: Path, filename: str, payload: Any, *, ensure_ascii: bool = False, indent: int = 2) -> None`
      - `_load_json(path: Path, filename: str) -> Any`
      - `_save_text(path: Path, filename: str, text: str) -> None`
      - `_load_text(path: Path, filename: str) -> str`
      - `_save_meta(path: Path, meta: Mapping[str, Any]) -> None`
      - `_load_meta(path: Path) -> dict[str, Any]`
      - `_save_pickle(path: Path, filename: str, obj: Any) -> None`
      - `_load_pickle(path: Path, filename: str) -> Any`
      - `_save_torch(path: Path, filename: str, payload: Any) -> None`
      - `_load_torch(path: Path, filename: str, *, map_location: str = 'cpu') -> Any`
      - `_common_meta(self, *, artifact_type: str | None = None, **extra: Any) -> dict[str, Any]`
- Top-level functions: 0

#### Module `core.artifacts.sklearn_mlp_artifact`
- File: `core\artifacts\sklearn_mlp_artifact.py`
- Top-level classes: 1
  - `class SklearnMLPSurrogateArtifact(ArtifactPersistenceBase)`
    - fields:
      - `artifact_id: str`
      - `model: Any`
      - `x_mean: np.ndarray`
      - `x_std: np.ndarray`
      - `residual_std: np.ndarray`
      - `feature_names: Sequence[str]`
      - `target_names: Sequence[str]`
      - `pipeline_name: str = 'identity'`
      - `pipeline_state: Dict[str, Any] = field(default_factory=dict)`
      - `ood_z_threshold: float = 4.0`
      - `metadata: Dict[str, Any] = field(default_factory=dict)`
    - methods:
      - `_transform(self, X: np.ndarray) -> np.ndarray`
      - `predict(self, X: np.ndarray) -> np.ndarray`
      - `uncertainty(self, X: np.ndarray) -> np.ndarray`
      - `validity(self, X: np.ndarray) -> np.ndarray`
      - `save(self, out_dir: str) -> None`
      - `load(cls, out_dir: str) -> 'SklearnMLPSurrogateArtifact'`
- Top-level functions: 1
  - `_as_2d(arr: np.ndarray) -> np.ndarray`

#### Module `core.artifacts.symbolic_artifact`
- File: `core\artifacts\symbolic_artifact.py`
- Top-level classes: 1
  - `class SymbolicSurrogateArtifact(ArtifactPersistenceBase)`
    - fields:
      - `artifact_id: str`
      - `genome: Sequence[Mapping[str, Any]]`
      - `parameter_values: Mapping[str, float]`
      - `readout_weight: np.ndarray`
      - `readout_bias: np.ndarray`
      - `x_mean: np.ndarray`
      - `x_std: np.ndarray`
      - `residual_std: np.ndarray`
      - `feature_names: Sequence[str]`
      - `target_names: Sequence[str]`
      - `pipeline_name: str = 'identity'`
      - `pipeline_state: Dict[str, Any] = field(default_factory=dict)`
      - `ood_z_threshold: float = 4.0`
      - `epsilon: float = 1e-06`
      - `metadata: Dict[str, Any] = field(default_factory=dict)`
    - methods:
      - `_transform(self, X: np.ndarray) -> np.ndarray`
      - `_basis(self, X: np.ndarray) -> np.ndarray`
      - `predict(self, X: np.ndarray) -> np.ndarray`
      - `uncertainty(self, X: np.ndarray) -> np.ndarray`
      - `validity(self, X: np.ndarray) -> np.ndarray`
      - `expression(self, *, target_index: int = 0, precision: int = 6, use_feature_names: bool = False) -> str`
      - `expressions(self, *, precision: int = 6, use_feature_names: bool = False) -> dict[str, str]`
      - `save(self, out_dir: str) -> None`
      - `load(cls, out_dir: str) -> 'SymbolicSurrogateArtifact'`
- Top-level functions: 2
  - `_as_2d(arr: np.ndarray) -> np.ndarray`
  - `_replace_feature_tokens(expr: str, feature_names: Sequence[str]) -> str`

#### Module `core.artifacts.symbolic_interval_artifact`
- File: `core\artifacts\symbolic_interval_artifact.py`
- Top-level classes: 1
  - `class SymbolicIntervalSurrogateArtifact(ArtifactPersistenceBase)`
    - fields:
      - `artifact_id: str`
      - `lower_quantile: float`
      - `upper_quantile: float`
      - `genome_low: Sequence[Mapping[str, Any]]`
      - `parameter_values_low: Mapping[str, float]`
      - `readout_weight_low: np.ndarray`
      - `readout_bias_low: np.ndarray`
      - `genome_high: Sequence[Mapping[str, Any]]`
      - `parameter_values_high: Mapping[str, float]`
      - `readout_weight_high: np.ndarray`
      - `readout_bias_high: np.ndarray`
      - `x_mean: np.ndarray`
      - `x_std: np.ndarray`
      - `residual_std: np.ndarray`
      - `calibration_margin: np.ndarray`
      - `feature_names: Sequence[str]`
      - `target_names: Sequence[str]`
      - `pipeline_name: str = 'identity'`
      - `pipeline_state: Dict[str, Any] = field(default_factory=dict)`
      - `ood_z_threshold: float = 4.0`
      - `epsilon: float = 1e-06`
      - `metadata: Dict[str, Any] = field(default_factory=dict)`
    - methods:
      - `_transform(self, X: np.ndarray) -> np.ndarray`
      - `_basis_low(self, X: np.ndarray) -> np.ndarray`
      - `_basis_high(self, X: np.ndarray) -> np.ndarray`
      - `predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
      - `predict(self, X: np.ndarray) -> np.ndarray`
      - `uncertainty(self, X: np.ndarray) -> np.ndarray`
      - `validity(self, X: np.ndarray) -> np.ndarray`
      - `expression(self, *, bound: str = 'center', target_index: int = 0, precision: int = 6, use_feature_names: bool = False) -> str`
      - `save(self, out_dir: str) -> None`
      - `load(cls, out_dir: str) -> 'SymbolicIntervalSurrogateArtifact'`
- Top-level functions: 3
  - `_as_2d(arr: np.ndarray) -> np.ndarray`
  - `_replace_feature_tokens(expr: str, feature_names: Sequence[str]) -> str`
  - `_expression_from_parts(genome: Sequence[Mapping[str, Any]], parameter_values: Mapping[str, float], readout_weight: np.ndarray, readout_bias: np.ndarray, *, target_index: int, precision: int, feature_names: Sequence[str], use_feature_names: bool) -> str`

#### Module `core.artifacts.torch_artifact`
- File: `core\artifacts\torch_artifact.py`
- Top-level classes: 1
  - `class TorchMLPSurrogateArtifact(ArtifactPersistenceBase)`
    - fields:
      - `artifact_id: str`
      - `input_dim: int`
      - `output_dim: int`
      - `hidden_dims: Sequence[int]`
      - `activation: str`
      - `dropout: float`
      - `model_state: Mapping[str, Any]`
      - `x_mean: np.ndarray`
      - `x_std: np.ndarray`
      - `residual_std: np.ndarray`
      - `feature_names: Sequence[str]`
      - `target_names: Sequence[str]`
      - `pipeline_name: str = 'identity'`
      - `pipeline_state: Dict[str, Any] = field(default_factory=dict)`
      - `ood_z_threshold: float = 4.0`
      - `metadata: Dict[str, Any] = field(default_factory=dict)`
      - `_model: Any = field(default=None, init=False, repr=False)`
    - methods:
      - `_transform(self, X: np.ndarray) -> np.ndarray`
      - `_get_model(self)`
      - `predict(self, X: np.ndarray) -> np.ndarray`
      - `uncertainty(self, X: np.ndarray) -> np.ndarray`
      - `validity(self, X: np.ndarray) -> np.ndarray`
      - `save(self, out_dir: str) -> None`
      - `load(cls, out_dir: str) -> 'TorchMLPSurrogateArtifact'`
- Top-level functions: 1
  - `_as_2d(arr: np.ndarray) -> np.ndarray`

#### Module `core.artifacts.xgboost_artifact`
- File: `core\artifacts\xgboost_artifact.py`
- Top-level classes: 1
  - `class XGBoostSurrogateArtifact(ArtifactPersistenceBase)`
    - fields:
      - `artifact_id: str`
      - `model: Any`
      - `x_mean: np.ndarray`
      - `x_std: np.ndarray`
      - `residual_std: np.ndarray`
      - `feature_names: Sequence[str]`
      - `target_names: Sequence[str]`
      - `pipeline_name: str = 'identity'`
      - `pipeline_state: Dict[str, Any] = field(default_factory=dict)`
      - `ood_z_threshold: float = 4.0`
      - `metadata: Dict[str, Any] = field(default_factory=dict)`
    - methods:
      - `_transform(self, X: np.ndarray) -> np.ndarray`
      - `predict(self, X: np.ndarray) -> np.ndarray`
      - `uncertainty(self, X: np.ndarray) -> np.ndarray`
      - `validity(self, X: np.ndarray) -> np.ndarray`
      - `save(self, out_dir: str) -> None`
      - `load(cls, out_dir: str) -> 'XGBoostSurrogateArtifact'`
- Top-level functions: 1
  - `_as_2d(arr: np.ndarray) -> np.ndarray`

#### Module `core.common.__init__`
- File: `core\common\__init__.py`
- `__all__`: `BaseSurrogateTrainer`, `BatchStream`, `BatchStreamSpec`, `create_torch_batch_stream`, `Cell`, `Sample`, `SampleDataset`, `ProcessedDataset`, `SurrogateArtifact`, `HypothesisSpace`, `TorchModuleHypothesisSpace`, `TrainingObjective`, `MSEObjective`, `PinballObjective`, `OptimizerSpec`, `create_regression_objective`, `create_quantile_objective`, `create_torch_optimizer`, `PreparedTrainingData`, `prepare_training_data`, `resolve_feature_target_names`, `resolve_torch_device`, `set_torch_seed`, `split_train_val_indices`
- Top-level classes: 0
- Top-level functions: 0

#### Module `core.common.base_trainer`
- File: `core\common\base_trainer.py`
- Top-level classes: 1
  - `class BaseSurrogateTrainer(ABC)`
    - methods:
      - `fit(self, data: ProcessedDataset | SampleDataset) -> SurrogateArtifact`
      - `capabilities(self) -> Dict[str, Any]`
- Top-level functions: 0

#### Module `core.common.batch_stream`
- File: `core\common\batch_stream.py`
- Top-level classes: 2
  - `class BatchStream(Protocol)`
    - methods:
      - `__iter__(self)`
      - `__len__(self) -> int`
  - `class BatchStreamSpec`
    - fields:
      - `batch_size: int = 64`
      - `shuffle: bool = True`
      - `drop_last: bool = False`
      - `num_workers: int = 0`
      - `pin_memory: bool = False`
- Top-level functions: 1
  - `create_torch_batch_stream(tensors: Sequence[Any], *, spec: BatchStreamSpec)`

#### Module `core.common.contracts`
- File: `core\common\contracts.py`
- Top-level classes: 5
  - `class ProcessedDataset`
    - fields:
      - `X_train: np.ndarray`
      - `y_train: np.ndarray`
      - `X_valid: np.ndarray | None = None`
      - `y_valid: np.ndarray | None = None`
      - `X_test: np.ndarray | None = None`
      - `y_test: np.ndarray | None = None`
      - `feature_names: Sequence[str] | None = None`
      - `target_names: Sequence[str] | None = None`
      - `metadata: Mapping[str, Any] | None = None`
  - `class Cell`
    - fields:
      - `name: str`
      - `payload: Any`
      - `modality: str = 'value'`
      - `labels: Mapping[str, Any] = field(default_factory=dict)`
      - `meta: Mapping[str, Any] = field(default_factory=dict)`
  - `class Sample`
    - fields:
      - `sample_id: str`
      - `cells: Mapping[str, Cell]`
      - `labels: Mapping[str, Any] = field(default_factory=dict)`
      - `meta: Mapping[str, Any] = field(default_factory=dict)`
  - `class SampleDataset`
    - fields:
      - `samples: Sequence[Sample]`
      - `target_key: str = 'target'`
      - `feature_cell_keys: Sequence[str] | None = None`
      - `target_names: Sequence[str] | None = None`
      - `description: str | None = None`
  - `class SurrogateArtifact(Protocol)`
    - fields:
      - `artifact_id: str`
      - `feature_names: Sequence[str]`
      - `target_names: Sequence[str]`
      - `metadata: Dict[str, Any]`
    - methods:
      - `predict(self, X: np.ndarray) -> np.ndarray`
      - `uncertainty(self, X: np.ndarray) -> np.ndarray`
      - `validity(self, X: np.ndarray) -> np.ndarray`
      - `save(self, out_dir: str) -> None`
- Top-level functions: 0

#### Module `core.common.hypothesis_space`
- File: `core\common\hypothesis_space.py`
- Top-level classes: 2
  - `class HypothesisSpace(Protocol)`
    - fields:
      - `name: str`
    - methods:
      - `forward(self, X: Any) -> Any`
      - `parameters(self) -> Iterable[Any]`
  - `class TorchModuleHypothesisSpace`
    - fields:
      - `module: Any`
      - `family: str`
      - `name: str = 'torch_module'`
    - methods:
      - `forward(self, X: Any) -> Any`
      - `parameters(self) -> Iterable[Any]`
- Top-level functions: 0

#### Module `core.common.loss_objective`
- File: `core\common\loss_objective.py`
- Top-level classes: 3
  - `class TrainingObjective(Protocol)`
    - fields:
      - `name: str`
    - methods:
      - `loss(self, pred: Any, target: Any, *, sample_weight: Any | None = None) -> Any`
  - `class MSEObjective`
    - fields:
      - `name: str = 'mse'`
    - methods:
      - `loss(self, pred, target, *, sample_weight = None)`
  - `class PinballObjective`
    - fields:
      - `quantile: float`
      - `name: str = 'pinball'`
    - methods:
      - `__post_init__(self) -> None`
      - `loss(self, pred, target, *, sample_weight = None)`
- Top-level functions: 4
  - `_ensure_torch() -> None`
  - `_weighted_reduce(per_sample_loss, sample_weight)`
  - `create_regression_objective(name: str = 'mse') -> TrainingObjective`
  - `create_quantile_objective(name: str = 'pinball', *, quantile: float) -> TrainingObjective`

#### Module `core.common.param_optimizer`
- File: `core\common\param_optimizer.py`
- Top-level classes: 1
  - `class OptimizerSpec`
    - fields:
      - `key: str = 'adamw'`
      - `params: Mapping[str, Any] = field(default_factory=dict)`
- Top-level functions: 2
  - `_ensure_torch() -> None`
  - `create_torch_optimizer(parameters: Iterable[Any], *, spec: OptimizerSpec, lr: float, weight_decay: float)`

#### Module `core.common.trainer_shared`
- File: `core\common\trainer_shared.py`
- Top-level classes: 1
  - `class PreparedTrainingData`
    - fields:
      - `normalized: ProcessedDataset`
      - `X: np.ndarray`
      - `Y: np.ndarray`
      - `context: Any`
      - `n: int`
      - `d: int`
      - `m: int`
      - `feature_names: tuple[str, ...]`
      - `target_names: tuple[str, ...]`
- Top-level functions: 5
  - `resolve_feature_target_names(normalized: ProcessedDataset, *, d: int, m: int) -> tuple[tuple[str, ...], tuple[str, ...]]`
  - `prepare_training_data(*, data: ProcessedDataset | SampleDataset, numericizer: Any, pipeline: Any, biases: Sequence[Any], fit_context_cls: Any) -> PreparedTrainingData`
  - `split_train_val_indices(n: int, *, val_ratio: float, seed: int, min_no_val_below: int = 10) -> tuple[np.ndarray, np.ndarray]`
  - `resolve_torch_device(torch_module: Any, requested: str) -> Any`
  - `set_torch_seed(torch_module: Any, seed: int) -> None`

#### Module `core.models.__init__`
- File: `core\models\__init__.py`
- `__all__`: `TorchMLPRegressor`, `SymbolicTorchRegressor`
- Top-level classes: 0
- Top-level functions: 0

#### Module `core.models.symbolic_torch_model`
- File: `core\models\symbolic_torch_model.py`
- Top-level classes: 1
  - `class SymbolicTorchRegressor(nn.Module)`
    - methods:
      - `__init__(self, input_dim: int, output_dim: int, *, genome: Sequence[Mapping[str, Any]], epsilon: float = 1e-06) -> None`
      - `parameter_specs(self) -> tuple[ParameterSpec, ...]`
      - `_parameter_values(self) -> dict[str, Any]`
      - `basis(self, X)`
      - `forward(self, X)`
      - `export_parameter_values(self) -> Dict[str, float]`
      - `export_readout(self) -> tuple[Any, Any]`
      - `expression_strings(self, *, with_values: bool = True) -> tuple[str, ...]`
- Top-level functions: 0

#### Module `core.models.torch_model`
- File: `core\models\torch_model.py`
- Top-level classes: 1
  - `class TorchMLPRegressor(nn.Module)`
    - methods:
      - `__init__(self, input_dim: int, output_dim: int, *, hidden_dims: Sequence[int] = (128, 64), activation: str = 'relu', dropout: float = 0.0) -> None`
      - `forward(self, x)`
- Top-level functions: 1
  - `_activation(name: str) -> nn.Module`

#### Module `core.orchestration.__init__`
- File: `core\orchestration\__init__.py`
- `__all__`: `BaseDataReader`, `MemoryDataReader`, `TrainDataBundle`, `TrainFlowSpec`, `SemanticTrainFlowSpec`, `TrainFlowResult`, `run_train_flow`, `run_semantic_train_flow`
- Top-level classes: 0
- Top-level functions: 0

#### Module `core.orchestration.workflow`
- File: `core\orchestration\workflow.py`
- Top-level classes: 6
  - `class TrainDataBundle`
    - fields:
      - `train: ProcessedDataset | SampleDataset`
      - `valid: ProcessedDataset | SampleDataset | None = None`
      - `test: ProcessedDataset | SampleDataset | None = None`
      - `metadata: Mapping[str, Any] | None = None`
  - `class BaseDataReader(Protocol)`
    - methods:
      - `read(self) -> TrainDataBundle`
  - `class MemoryDataReader`
    - fields:
      - `bundle: TrainDataBundle`
    - methods:
      - `read(self) -> TrainDataBundle`
  - `class TrainFlowSpec`
    - fields:
      - `assembly: TrainerAssemblySpec`
      - `eval_splits: Sequence[str] = ('train', 'valid', 'test')`
      - `output_dir: str | None = None`
      - `save_artifact: bool = True`
      - `save_report: bool = True`
      - `run_name: str = 'train_flow'`
  - `class SemanticTrainFlowSpec`
    - fields:
      - `assembly: FlowAssemblySpec = field(default_factory=FlowAssemblySpec)`
      - `eval_splits: Sequence[str] = ('train', 'valid', 'test')`
      - `output_dir: str | None = None`
      - `save_artifact: bool = True`
      - `save_report: bool = True`
      - `run_name: str = 'semantic_train_flow'`
  - `class TrainFlowResult`
    - fields:
      - `artifact: SurrogateArtifact`
      - `processed: ProcessedDataset`
      - `metrics: Dict[str, Dict[str, float]]`
      - `report: Dict[str, Any]`
      - `output_dir: str | None = None`
- Top-level functions: 11
  - `_merge_metadata(*parts: Mapping[str, Any] | None) -> Dict[str, Any]`
  - `_jsonable(value: Any) -> Any`
  - `_is_finite_matrix(arr: np.ndarray) -> bool`
  - `_evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]`
  - `_normalize_eval_splits(splits: Sequence[str]) -> tuple[str, ...]`
  - `_encode_sampleset(dataset: SampleDataset, *, numericizer: BaseNumericizer, split_name: str) -> tuple[np.ndarray, np.ndarray]`
  - `_to_processed_bundle(bundle: TrainDataBundle, *, numericizer: BaseNumericizer | None = None) -> tuple[ProcessedDataset, BaseNumericizer | None]`
  - `_collect_eval_pairs(processed: ProcessedDataset) -> Dict[str, tuple[np.ndarray, np.ndarray]]`
  - `_build_report(*, spec: TrainFlowSpec, processed: ProcessedDataset, metrics: Mapping[str, Mapping[str, float]], artifact: SurrogateArtifact, trainer_name: str) -> Dict[str, Any]`
  - `run_train_flow(data: TrainDataBundle | BaseDataReader | ProcessedDataset | SampleDataset, *, spec: TrainFlowSpec, numericizer: BaseNumericizer | None = None)`
  - `run_semantic_train_flow(data: TrainDataBundle | BaseDataReader | ProcessedDataset | SampleDataset, *, spec: SemanticTrainFlowSpec, config: Any | None = None) -> TrainFlowResult`

#### Module `core.symbolic.__init__`
- File: `core\symbolic\__init__.py`
- `__all__`: `GradientSignal`, `GradientParser`, `GradientCorrectionConfig`, `GradientCorrection`, `PathPrior`, `SymbolicPathMemory`, `default_path_memory_db`, `StructureScoreConfig`, `StructureOptimizer`, `differentiate_expression_wrt_param`, `differentiate_expression_wrt_feature`, `gradient_formula_strings`, `evaluate_gradient_numpy`, `StructureSearchConfig`, `StructureSearchResult`, `residual_guided_structure_search`, `evaluate_genome_with_ridge`, `regression_metrics`
- Top-level classes: 0
- Top-level functions: 0

#### Module `core.symbolic.gradient_correction`
- File: `core\symbolic\gradient_correction.py`
- `__all__`: `GradientCorrectionConfig`, `GradientCorrection`
- Top-level classes: 2
  - `class GradientCorrectionConfig`
    - fields:
      - `focus_topk_features: int = 3`
      - `min_priority: float = 0.0001`
  - `class GradientCorrection`
    - methods:
      - `__init__(self, signal: GradientSignal, *, config: GradientCorrectionConfig | None = None) -> None`
      - `_build_active_feature_index(self) -> tuple[int, ...]`
      - `active_features(self) -> tuple[int, ...]`
      - `score_candidate(self, *, expr: Mapping[str, Any], X: np.ndarray, coeff_vector: np.ndarray, feature_indices: Sequence[int] | None = None) -> dict[str, Any]`
- Top-level functions: 2
  - `_as_2d_float(arr: np.ndarray) -> np.ndarray`
  - `_safe_corr(a: np.ndarray, b: np.ndarray) -> float`

#### Module `core.symbolic.gradient_parser`
- File: `core\symbolic\gradient_parser.py`
- `__all__`: `GradientSignal`, `GradientParser`
- Top-level classes: 2
  - `class GradientSignal`
    - fields:
      - `overall_mismatch: float`
      - `feature_mismatch: np.ndarray`
      - `feature_priority: np.ndarray`
      - `feature_gap_signed_mean: np.ndarray`
      - `feature_gap_abs_mean: np.ndarray`
      - `feature_valid_fraction: np.ndarray`
      - `gap_by_feature: tuple[np.ndarray, ...]`
  - `class GradientParser`
    - methods:
      - `_local_slope_1d(x_col: np.ndarray, y_mat: np.ndarray) -> np.ndarray`
      - `_local_slope_binned_median(cls, x_col: np.ndarray, y_mat: np.ndarray, *, bins: int, min_bin_samples: int) -> np.ndarray`
      - `_local_slope(cls, x_col: np.ndarray, y_mat: np.ndarray, *, mode: str, bins: int, min_bin_samples: int) -> np.ndarray`
      - `model_partial_derivative(genome: Sequence[Mapping[str, Any]], weight: np.ndarray, X: np.ndarray, *, feature_index: int) -> np.ndarray`
      - `_nanmean_or_zero(arr: np.ndarray) -> float`
      - `build_signal(cls, *, genome: Sequence[Mapping[str, Any]], weight: np.ndarray, X: np.ndarray, y: np.ndarray, slope_mode: str = 'central_diff', slope_bins: int = 24, slope_min_bin_samples: int = 12) -> GradientSignal`
      - `gradient_mismatch(cls, *, genome: Sequence[Mapping[str, Any]], weight: np.ndarray, X: np.ndarray, y: np.ndarray, slope_mode: str = 'central_diff', slope_bins: int = 24, slope_min_bin_samples: int = 12) -> float`
- Top-level functions: 1
  - `_as_2d_float(arr: np.ndarray) -> np.ndarray`

#### Module `core.symbolic.path_memory`
- File: `core\symbolic\path_memory.py`
- `__all__`: `PathPrior`, `SymbolicPathMemory`, `default_path_memory_db`
- Top-level classes: 2
  - `class PathPrior`
    - fields:
      - `seen: int = 0`
      - `success: int = 0`
      - `failure: int = 0`
      - `total_delta_rmse: float = 0.0`
      - `total_selected_score: float = 0.0`
    - methods:
      - `outcomes(self) -> int`
      - `accept_rate(self) -> float`
      - `avg_delta_rmse(self) -> float`
      - `avg_selected_score(self) -> float`
      - `to_dict(self) -> dict[str, float | int]`
  - `class SymbolicPathMemory`
    - methods:
      - `__init__(self, *, db_path: str | None = None, namespace: str = 'global') -> None`
      - `_detect_backend(raw: str) -> str`
      - `_connect_mysql(dsn: str, pymysql_mod: Any)`
      - `genome_signature(expr_keys: Sequence[str]) -> str`
      - `close(self) -> None`
      - `_execute(self, query: str, params: Sequence[Any] = ()) -> Any`
      - `_execute_nonquery(self, query: str, params: Sequence[Any] = ()) -> None`
      - `_query_one(self, query: str, params: Sequence[Any] = ()) -> tuple[Any, ...] | None`
      - `_commit(self) -> None`
      - `_ensure_schema(self) -> None`
      - `get_expr_prior(self, expr_key: str) -> PathPrior`
      - `touch_expr(self, expr_key: str) -> None`
      - `record_expr_outcome(self, expr_key: str, *, selected_score: float, delta_rmse: float, success: bool) -> None`
      - `record_edge(self, *, src_sig: str, op: str, expr_key: str, dst_sig: str, delta_rmse: float, success: bool) -> None`
- Top-level functions: 3
  - `_utc_now() -> str`
  - `_expr_hash(expr_key: str) -> str`
  - `default_path_memory_db() -> Path`

#### Module `core.symbolic.structure_optimizer`
- File: `core\symbolic\structure_optimizer.py`
- `__all__`: `StructureScoreConfig`, `StructureOptimizer`
- Top-level classes: 2
  - `class StructureScoreConfig`
    - fields:
      - `score_corr_bonus: float = 0.04`
      - `score_complexity_penalty: float = 0.0007`
      - `score_grad_guidance_bonus: float = 0.0`
  - `class StructureOptimizer`
    - methods:
      - `__init__(self, config: StructureScoreConfig | None = None) -> None`
      - `combine(self, *, projected_gain: float, abs_corr: float, complexity: float, grad_alignment: float) -> dict[str, Any]`
- Top-level functions: 0

#### Module `core.symbolic.symbolic_dsl`
- File: `core\symbolic\symbolic_dsl.py`
- Top-level classes: 1
  - `class ParameterSpec`
    - fields:
      - `name: str`
      - `init: float = 1.0`
      - `trainable: bool = True`
- Top-level functions: 25
  - `default_genome(input_dim: int, *, ops: Sequence[str] = ('identity', 'square', 'sin', 'cos')) -> tuple[Dict[str, Any], ...]`
  - `_normalize_expression(expr: Mapping[str, Any], *, input_dim: int) -> Dict[str, Any]`
  - `normalize_genome(genome: Sequence[Mapping[str, Any]], *, input_dim: int) -> tuple[Dict[str, Any], ...]`
  - `collect_parameter_specs(genome: Sequence[Mapping[str, Any]]) -> tuple[ParameterSpec, ...]`
  - `_safe_div_numpy(a: np.ndarray, b: np.ndarray, *, eps: float) -> np.ndarray`
  - `_safe_log_numpy(x: np.ndarray, *, eps: float) -> np.ndarray`
  - `_safe_sqrt_numpy(x: np.ndarray, *, eps: float) -> np.ndarray`
  - `evaluate_expression_numpy(expr: Mapping[str, Any], X: np.ndarray, *, param_values: Mapping[str, float] | None = None, eps: float = 1e-06) -> np.ndarray`
  - `evaluate_genome_numpy(genome: Sequence[Mapping[str, Any]], X: np.ndarray, *, param_values: Mapping[str, float] | None = None, eps: float = 1e-06) -> np.ndarray`
  - `_safe_div_torch(a, b, *, eps: float)`
  - `_safe_log_torch(x, *, eps: float)`
  - `_safe_sqrt_torch(x, *, eps: float)`
  - `evaluate_expression_torch(expr: Mapping[str, Any], X, *, param_values: Mapping[str, Any] | None = None, eps: float = 1e-06)`
  - `evaluate_genome_torch(genome: Sequence[Mapping[str, Any]], X, *, param_values: Mapping[str, Any] | None = None, eps: float = 1e-06)`
  - `_fmt_scalar(v: float, precision: int = 6) -> str`
  - `expression_to_string(expr: Mapping[str, Any], *, param_values: Mapping[str, float] | None = None, precision: int = 6) -> str`
  - `genome_to_strings(genome: Sequence[Mapping[str, Any]], *, param_values: Mapping[str, float] | None = None, precision: int = 6) -> tuple[str, ...]`
  - `detect_binary_columns(X: np.ndarray, *, round_decimals: int = 10) -> tuple[bool, ...]`
  - `_feature_expr(index: int) -> Dict[str, Any]`
  - `_const_expr(value: float) -> Dict[str, Any]`
  - `_unary_expr(op: str, arg: Mapping[str, Any]) -> Dict[str, Any]`
  - `_binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]`
  - `_relu_expr(arg: Mapping[str, Any]) -> Dict[str, Any]`
  - `_feature_scores(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray`
  - `default_genome_v2(X: np.ndarray, *, y: np.ndarray | None = None, continuous_ops: Sequence[str] = ('identity', 'sin', 'cos'), binary_ops: Sequence[str] = ('identity',), include_interactions: bool = True, max_interactions: int = 20, topk_features: int = 6, include_hinge: bool = True, hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75)) -> tuple[Dict[str, Any], ...]`

#### Module `core.symbolic.symbolic_gradient`
- File: `core\symbolic\symbolic_gradient.py`
- `__all__`: `differentiate_expression_wrt_param`, `differentiate_expression_wrt_feature`, `gradient_formula_strings`, `evaluate_gradient_numpy`
- Top-level classes: 0
- Top-level functions: 19
  - `_is_const(node: Mapping[str, Any], value: float, *, tol: float = 1e-12) -> bool`
  - `_const(value: float) -> Dict[str, Any]`
  - `_copy_expr(node: Mapping[str, Any]) -> Dict[str, Any]`
  - `_unary(op: str, arg: Mapping[str, Any]) -> Dict[str, Any]`
  - `_binary(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]`
  - `_add(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]`
  - `_sub(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]`
  - `_mul(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]`
  - `_div(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]`
  - `_square(a: Mapping[str, Any]) -> Dict[str, Any]`
  - `_safe_abs(node: Mapping[str, Any]) -> Dict[str, Any]`
  - `_safe_sqrt(node: Mapping[str, Any]) -> Dict[str, Any]`
  - `_plus_eps(node: Mapping[str, Any], eps: float) -> Dict[str, Any]`
  - `_d_expr_wrt_param(node: Mapping[str, Any], *, param_name: str, eps: float) -> Dict[str, Any]`
  - `_d_expr_wrt_feature(node: Mapping[str, Any], *, feature_index: int, eps: float) -> Dict[str, Any]`
  - `differentiate_expression_wrt_param(expr: Mapping[str, Any], *, param_name: str, eps: float = 1e-06) -> Dict[str, Any]`
  - `differentiate_expression_wrt_feature(expr: Mapping[str, Any], *, feature_index: int, eps: float = 1e-06) -> Dict[str, Any]`
  - `gradient_formula_strings(expr: Mapping[str, Any], *, param_names: Sequence[str] = (), feature_indices: Sequence[int] = (), param_values: Mapping[str, float] | None = None, eps: float = 1e-06, precision: int = 6) -> Dict[str, str]`
  - `evaluate_gradient_numpy(expr: Mapping[str, Any], X: np.ndarray, *, param_name: str | None = None, feature_index: int | None = None, param_values: Mapping[str, float] | None = None, eps: float = 1e-06) -> np.ndarray`

#### Module `core.symbolic.symbolic_structure_search`
- File: `core\symbolic\symbolic_structure_search.py`
- `__all__`: `StructureSearchConfig`, `StructureSearchResult`, `residual_guided_structure_search`, `evaluate_genome_with_ridge`, `regression_metrics`
- Top-level classes: 2
  - `class StructureSearchConfig`
    - fields:
      - `max_added_terms: int = 10`
      - `topk_features: int = 8`
      - `max_pair_terms: int = 16`
      - `max_candidates_per_iter: int = 500`
      - `candidate_keep_top: int = 12`
      - `ridge_l2: float = 0.0001`
      - `min_score: float = 1e-06`
      - `min_projected_gain: float = 1e-07`
      - `score_complexity_penalty: float = 0.0007`
      - `score_corr_bonus: float = 0.04`
      - `score_grad_guidance_bonus: float = 0.08`
      - `min_actual_rmse_gain: float = 0.0`
      - `overfit_guard_enabled: bool = False`
      - `overfit_guard_val_ratio: float = 0.2`
      - `overfit_guard_min_val_samples: int = 64`
      - `overfit_guard_random_seed: int = 42`
      - `overfit_guard_min_val_rmse_gain: float = 0.0`
      - `overfit_guard_max_gap_increase: float = 0.05`
      - `overfit_guard_patience: int = 3`
      - `overfit_guard_snapshot_min_improve: float = 0.0`
      - `overfit_guard_tabu_rounds: int = 2`
      - `overfit_guard_replace_topk: int = 3`
      - `overfit_guard_replace_drop_topk: int = 3`
      - `grad_focus_topk: int = 3`
      - `grad_min_priority: float = 0.0001`
      - `grad_slope_mode: str = 'central_diff'`
      - `grad_slope_bins: int = 24`
      - `grad_slope_min_bin_samples: int = 12`
      - `grad_adv_check: bool = False`
      - `grad_adv_trials: int = 3`
      - `grad_adv_noise_std: float = 0.02`
      - `grad_adv_min_stability: float = 0.0`
      - `grad_adv_random_seed: int = 42`
      - `include_hinge: bool = True`
      - `hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75)`
      - `unary_ops: Sequence[str] = ('square', 'sin', 'cos', 'tanh')`
      - `nested_unary_patterns: Sequence[str] = ('sin(square)', 'cos(square)')`
      - `max_arity: int = 3`
      - `max_expr_depth: int = 8`
      - `enable_grad_residual_projection: bool = True`
      - `grad_projection_topk_focus: int = 3`
      - `grad_projection_partner_pool: int = 8`
      - `grad_projection_topk_partners: int = 3`
      - `grad_projection_topk_unary: int = 2`
      - `grad_projection_partner_orders: Sequence[int] = (1, 2)`
      - `grad_projection_enable_pair_dictionary: bool = True`
      - `grad_projection_min_abs_corr: float = 0.05`
      - `grad_projection_max_generated: int = 120`
      - `enable_prune: bool = True`
      - `prune_rmse_tolerance: float = 1e-08`
      - `prune_max_removed_per_iter: int = 1`
      - `path_memory_enabled: bool = True`
      - `path_memory_db_path: str = ''`
      - `path_memory_namespace: str = 'global'`
      - `path_memory_prior_bonus: float = 0.03`
      - `path_memory_tabu_penalty: float = 0.06`
      - `path_memory_min_outcomes: int = 3`
      - `path_memory_hard_tabu: bool = False`
      - `path_memory_hard_tabu_accept_rate: float = 0.1`
  - `class StructureSearchResult`
    - fields:
      - `genome: tuple[Dict[str, Any], ...]`
      - `base_metrics: dict[str, float]`
      - `final_metrics: dict[str, float]`
      - `iterations: tuple[dict[str, Any], ...]`
      - `weight: np.ndarray`
      - `bias: np.ndarray`
      - `score_trace: tuple[float, ...]`
    - methods:
      - `to_dict(self) -> dict[str, Any]`
- Top-level functions: 34
  - `_as_2d_float(arr: np.ndarray) -> np.ndarray`
  - `_feature_expr(index: int) -> Dict[str, Any]`
  - `_const_expr(value: float) -> Dict[str, Any]`
  - `_unary_expr(op: str, arg: Mapping[str, Any]) -> Dict[str, Any]`
  - `_binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]`
  - `_relu_expr(arg: Mapping[str, Any]) -> Dict[str, Any]`
  - `_expr_key(expr: Mapping[str, Any]) -> str`
  - `_expr_node_count(expr: Mapping[str, Any]) -> int`
  - `_expr_depth(expr: Mapping[str, Any]) -> int`
  - `_expr_features(expr: Mapping[str, Any]) -> tuple[int, ...]`
  - `_safe_corr(a: np.ndarray, b: np.ndarray) -> float`
  - `_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`
  - `_fit_ridge_readout(phi: np.ndarray, y: np.ndarray, *, l2: float) -> dict[str, np.ndarray]`
  - `_design_matrix_from_genome(genome: Sequence[Mapping[str, Any]], X: np.ndarray) -> np.ndarray`
  - `_feature_residual_scores(X: np.ndarray, residual: np.ndarray) -> np.ndarray`
  - `_default_seed_genome(input_dim: int) -> tuple[Dict[str, Any], ...]`
  - `_genome_expr_keys(genome: Sequence[Mapping[str, Any]]) -> tuple[str, ...]`
  - `_genome_signature(genome: Sequence[Mapping[str, Any]]) -> str`
  - `_build_nested_expr(pattern: str, base: Mapping[str, Any]) -> Dict[str, Any] | None`
  - `_safe_corr_masked(a: np.ndarray, b: np.ndarray) -> float`
  - `_candidate_allowed(expr: Mapping[str, Any], *, cfg: StructureSearchConfig) -> bool`
  - `_feature_transform_library(*, X: np.ndarray, feature_idx: int, cfg: StructureSearchConfig, is_binary: np.ndarray) -> list[dict[str, Any]]`
  - `_build_grad_projection_candidates(*, X: np.ndarray, cfg: StructureSearchConfig, gradient_signal: Any | None, residual_selected: Sequence[int], is_binary: np.ndarray) -> list[dict[str, Any]]`
  - `_finalize_candidates(candidates: Sequence[Mapping[str, Any]], *, cfg: StructureSearchConfig) -> list[dict[str, Any]]`
  - `_build_candidates(X: np.ndarray, residual: np.ndarray, *, cfg: StructureSearchConfig, gradient_signal: Any | None = None) -> list[dict[str, Any]]`
  - `_score_candidate(candidate: Mapping[str, Any], *, X: np.ndarray, residual: np.ndarray, cfg: StructureSearchConfig, gradient_correction: GradientCorrection | None = None, structure_optimizer: StructureOptimizer | None = None, x_scale: np.ndarray | None = None, grad_adv_config: Mapping[str, Any] | None = None, rng: np.random.Generator | None = None, path_memory: SymbolicPathMemory | None = None) -> dict[str, Any] | None`
  - `evaluate_genome_with_ridge(genome: Sequence[Mapping[str, Any]], *, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray | None = None, y_eval: np.ndarray | None = None, l2: float = 0.0001) -> dict[str, Any]`
  - `_candidate_log_row(item: Mapping[str, Any]) -> dict[str, Any]`
  - `_fit_with_metrics(genome: Sequence[Mapping[str, Any]], *, X: np.ndarray, y: np.ndarray, l2: float) -> tuple[dict[str, np.ndarray], dict[str, float]]`
  - `_predict_with_fit(genome: Sequence[Mapping[str, Any]], *, X_eval: np.ndarray, fit: Mapping[str, np.ndarray]) -> np.ndarray`
  - `_split_fit_val(X: np.ndarray, y: np.ndarray, *, val_ratio: float, min_val_samples: int, random_seed: int) -> dict[str, Any]`
  - `_prune_terms_once(genome: list[Dict[str, Any]], *, X: np.ndarray, y: np.ndarray, cfg: StructureSearchConfig) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, float]]`
  - `residual_guided_structure_search(X: np.ndarray, y: np.ndarray, *, feature_names: Sequence[str] | None = None, seed_genome: Sequence[Mapping[str, Any]] | None = None, config: StructureSearchConfig | None = None) -> StructureSearchResult`
  - `regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`

#### Module `core.trainers.__init__`
- File: `core\trainers\__init__.py`
- `__all__`: `RidgeTrainerConfig`, `RidgeSurrogateTrainer`, `TorchMLPTrainerConfig`, `TorchMLPSurrogateTrainer`, `SklearnMLPTrainerConfig`, `SklearnMLPSurrogateTrainer`, `XGBoostTrainerConfig`, `XGBoostSurrogateTrainer`, `SymbolicTorchTrainerConfig`, `SymbolicTorchSurrogateTrainer`, `SymbolicTorchIntervalTrainerConfig`, `SymbolicTorchIntervalTrainer`, `SymbolicStagewiseTrainerConfig`, `SymbolicStagewiseSurrogateTrainer`
- Top-level classes: 0
- Top-level functions: 0

#### Module `core.trainers.sklearn_mlp_trainer`
- File: `core\trainers\sklearn_mlp_trainer.py`
- Top-level classes: 2
  - `class SklearnMLPTrainerConfig`
    - fields:
      - `artifact_id: str = 'sklearn_mlp_surrogate_v1'`
      - `hidden_layer_sizes: Sequence[int] = (128, 64)`
      - `activation: str = 'relu'`
      - `solver: str = 'adam'`
      - `alpha: float = 0.0001`
      - `batch_size: int | str = 'auto'`
      - `learning_rate_init: float = 0.001`
      - `max_iter: int = 300`
      - `tol: float = 0.0001`
      - `n_iter_no_change: int = 20`
      - `validation_fraction: float = 0.15`
      - `early_stopping: bool = True`
      - `random_seed: int = 42`
      - `ood_z_threshold: float = 4.0`
      - `verbose: bool = False`
  - `class SklearnMLPSurrogateTrainer(BaseSurrogateTrainer)`
    - methods:
      - `__init__(self, config: SklearnMLPTrainerConfig | None = None, *, pipeline: BasePipeline | None = None, biases: Sequence[BaseTrainingBias] | None = None, numericizer: BaseNumericizer | None = None, modality_encoders: Mapping[str, ModalityEncoder] | None = None, target_codecs: Mapping[str, TargetCodec] | None = None, target_codec: str | None = None, categorical_unknown: str = 'error') -> None`
      - `capabilities(self) -> dict[str, object]`
      - `fit(self, data: ProcessedDataset | SampleDataset) -> SklearnMLPSurrogateArtifact`
- Top-level functions: 0

#### Module `core.trainers.symbolic_stagewise_trainer`
- File: `core\trainers\symbolic_stagewise_trainer.py`
- Top-level classes: 2
  - `class SymbolicStagewiseTrainerConfig`
    - fields:
      - `artifact_id: str = 'symbolic_stagewise_surrogate_v1'`
      - `force_linear_base: str | bool = 'on'`
      - `keep_search_trace: bool = True`
      - `auto_val_ratio: float = 0.2`
      - `auto_min_val_samples: int = 64`
      - `auto_random_seed: int = 42`
      - `auto_term_penalty: float = 0.001`
      - `auto_depth_penalty: float = 0.002`
      - `auto_grad_penalty: float = 0.05`
      - `search_max_added_terms: int = 10`
      - `search_topk_features: int = 8`
      - `search_max_pair_terms: int = 16`
      - `search_max_candidates_per_iter: int = 500`
      - `search_candidate_keep_top: int = 12`
      - `search_max_arity: int = 3`
      - `search_max_expr_depth: int = 8`
      - `search_min_actual_rmse_gain: float = 0.0`
      - `search_overfit_guard_enabled: bool = False`
      - `search_overfit_guard_val_ratio: float = 0.2`
      - `search_overfit_guard_min_val_samples: int = 64`
      - `search_overfit_guard_random_seed: int = 42`
      - `search_overfit_guard_min_val_rmse_gain: float = 0.0`
      - `search_overfit_guard_max_gap_increase: float = 0.05`
      - `search_overfit_guard_patience: int = 3`
      - `search_overfit_guard_snapshot_min_improve: float = 0.0`
      - `search_overfit_guard_tabu_rounds: int = 2`
      - `search_overfit_guard_replace_topk: int = 3`
      - `search_overfit_guard_replace_drop_topk: int = 3`
      - `search_ridge_l2: float = 0.0001`
      - `search_min_score: float = 1e-06`
      - `search_min_projected_gain: float = 1e-07`
      - `search_score_complexity_penalty: float = 0.0007`
      - `search_score_corr_bonus: float = 0.04`
      - `search_grad_guidance_bonus: float = 0.08`
      - `search_grad_focus_topk: int = 3`
      - `search_grad_min_priority: float = 0.0001`
      - `search_grad_slope_mode: str = 'central_diff'`
      - `search_grad_slope_bins: int = 24`
      - `search_grad_slope_min_bin_samples: int = 12`
      - `search_grad_adv_check: bool = False`
      - `search_grad_adv_trials: int = 3`
      - `search_grad_adv_noise_std: float = 0.02`
      - `search_grad_adv_min_stability: float = 0.0`
      - `search_grad_adv_random_seed: int = 42`
      - `search_include_hinge: bool = True`
      - `search_hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75)`
      - `search_unary_ops: Sequence[str] = ('square', 'sin', 'cos', 'tanh')`
      - `search_nested_unary_patterns: Sequence[str] = ('sin(square)', 'cos(square)')`
      - `search_enable_grad_residual_projection: bool = True`
      - `search_grad_projection_topk_focus: int = 3`
      - `search_grad_projection_partner_pool: int = 8`
      - `search_grad_projection_topk_partners: int = 3`
      - `search_grad_projection_topk_unary: int = 2`
      - `search_grad_projection_partner_orders: Sequence[int] = (1, 2)`
      - `search_grad_projection_enable_pair_dictionary: bool = True`
      - `search_grad_projection_min_abs_corr: float = 0.05`
      - `search_grad_projection_max_generated: int = 120`
      - `search_enable_prune: bool = True`
      - `search_prune_rmse_tolerance: float = 1e-08`
      - `search_prune_max_removed_per_iter: int = 1`
      - `search_path_memory_enabled: bool = True`
      - `search_path_memory_db_path: str = ''`
      - `search_path_memory_namespace: str = 'global'`
      - `search_path_memory_prior_bonus: float = 0.03`
      - `search_path_memory_tabu_penalty: float = 0.06`
      - `search_path_memory_min_outcomes: int = 3`
      - `search_path_memory_hard_tabu: bool = False`
      - `search_path_memory_hard_tabu_accept_rate: float = 0.1`
      - `ood_z_threshold: float = 4.0`
      - `epsilon: float = 1e-06`
  - `class SymbolicStagewiseSurrogateTrainer(BaseSurrogateTrainer)`
    - methods:
      - `__init__(self, config: SymbolicStagewiseTrainerConfig | None = None, *, pipeline: BasePipeline | None = None, biases: Sequence[BaseTrainingBias] | None = None, numericizer: BaseNumericizer | None = None, modality_encoders: Mapping[str, ModalityEncoder] | None = None, target_codecs: Mapping[str, TargetCodec] | None = None, target_codec: str | None = None, categorical_unknown: str = 'error') -> None`
      - `capabilities(self) -> dict[str, object]`
      - `_build_search_config(self) -> StructureSearchConfig`
      - `_linear_seed_genome(input_dim: int) -> tuple[dict[str, Any], ...]`
      - `_normalize_linear_mode(value: str | bool) -> str`
      - `_seed_genome_by_mode(cls, mode: str, input_dim: int) -> tuple[dict[str, Any], ...]`
      - `_expr_depth(expr: Mapping[str, Any]) -> int`
      - `_genome_complexity(cls, genome: Sequence[Mapping[str, Any]]) -> dict[str, float]`
      - `_local_slope_1d(x_col: np.ndarray, y_mat: np.ndarray) -> np.ndarray`
      - `_model_partial_derivative(genome: Sequence[Mapping[str, Any]], weight: np.ndarray, X: np.ndarray, *, feature_index: int) -> np.ndarray`
      - `_gradient_mismatch_metric(self, *, genome: Sequence[Mapping[str, Any]], weight: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float`
      - `_fit_search_once(self, X: np.ndarray, Y: np.ndarray, *, feature_names: Sequence[str], mode: str, search_cfg: StructureSearchConfig) -> tuple[Any, dict[str, Any]]`
      - `_auto_select_mode(self, X: np.ndarray, Y: np.ndarray, *, feature_names: Sequence[str], search_cfg: StructureSearchConfig) -> tuple[str, dict[str, Any]]`
      - `fit(self, data: ProcessedDataset | SampleDataset) -> SymbolicSurrogateArtifact`
- Top-level functions: 0

#### Module `core.trainers.symbolic_torch_interval_trainer`
- File: `core\trainers\symbolic_torch_interval_trainer.py`
- Top-level classes: 2
  - `class SymbolicTorchIntervalTrainerConfig`
    - fields:
      - `artifact_id: str = 'symbolic_torch_interval_surrogate_v1'`
      - `version: str = 'v2'`
      - `lower_quantile: float = 0.1`
      - `upper_quantile: float = 0.9`
      - `genome: Sequence[Mapping[str, Any]] | None = None`
      - `library_ops: Sequence[str] = ('identity', 'square', 'sin', 'cos')`
      - `v2_continuous_ops: Sequence[str] = ('identity', 'sin', 'cos')`
      - `v2_binary_ops: Sequence[str] = ('identity',)`
      - `v2_include_interactions: bool = True`
      - `v2_max_interactions: int = 20`
      - `v2_topk_features: int = 6`
      - `v2_include_hinge: bool = True`
      - `v2_hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75)`
      - `order_penalty: float = 8.0`
      - `width_penalty: float = 0.0`
      - `l1_readout: float = 0.0`
      - `l1_params: float = 0.0`
      - `conformal_calibration: bool = True`
      - `conformal_level: float | None = None`
      - `epochs: int = 260`
      - `batch_size: int = 128`
      - `batch_shuffle: bool = True`
      - `batch_drop_last: bool = False`
      - `batch_num_workers: int = 0`
      - `batch_pin_memory: bool = False`
      - `lr: float = 0.001`
      - `weight_decay: float = 0.0001`
      - `optimizer: str = 'adamw'`
      - `optimizer_params: Mapping[str, Any] = field(default_factory=dict)`
      - `quantile_objective: str = 'pinball'`
      - `val_ratio: float = 0.15`
      - `early_stop_patience: int = 30`
      - `early_stop_min_delta: float = 1e-06`
      - `random_seed: int = 42`
      - `device: str = 'auto'`
      - `ood_z_threshold: float = 4.0`
      - `epsilon: float = 1e-06`
      - `verbose: bool = False`
  - `class SymbolicTorchIntervalTrainer(BaseSurrogateTrainer)`
    - methods:
      - `__init__(self, config: SymbolicTorchIntervalTrainerConfig | None = None, *, pipeline: BasePipeline | None = None, biases: Sequence[BaseTrainingBias] | None = None, numericizer: BaseNumericizer | None = None, modality_encoders: Mapping[str, ModalityEncoder] | None = None, target_codecs: Mapping[str, TargetCodec] | None = None, target_codec: str | None = None, categorical_unknown: str = 'error') -> None`
      - `capabilities(self) -> dict[str, object]`
      - `_resolve_device(self) -> torch.device`
      - `_split_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]`
      - `_build_genome(self, X_basis: np.ndarray, Y_basis: np.ndarray)`
      - `_interval_metrics(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, float]`
      - `_apply_margin(low: np.ndarray, high: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
      - `fit(self, data: ProcessedDataset | SampleDataset) -> SymbolicIntervalSurrogateArtifact`
- Top-level functions: 0

#### Module `core.trainers.symbolic_torch_trainer`
- File: `core\trainers\symbolic_torch_trainer.py`
- Top-level classes: 2
  - `class SymbolicTorchTrainerConfig`
    - fields:
      - `artifact_id: str = 'symbolic_torch_surrogate_v2'`
      - `version: str = 'v2'`
      - `genome: Sequence[Mapping[str, Any]] | None = None`
      - `library_ops: Sequence[str] = ('identity', 'square', 'sin', 'cos')`
      - `v2_continuous_ops: Sequence[str] = ('identity', 'sin', 'cos')`
      - `v2_binary_ops: Sequence[str] = ('identity',)`
      - `v2_include_interactions: bool = True`
      - `v2_max_interactions: int = 20`
      - `v2_topk_features: int = 6`
      - `v2_include_hinge: bool = True`
      - `v2_hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75)`
      - `epochs: int = 220`
      - `batch_size: int = 128`
      - `batch_shuffle: bool = True`
      - `batch_drop_last: bool = False`
      - `batch_num_workers: int = 0`
      - `batch_pin_memory: bool = False`
      - `lr: float = 0.001`
      - `weight_decay: float = 0.0001`
      - `optimizer: str = 'adamw'`
      - `optimizer_params: Mapping[str, Any] = field(default_factory=dict)`
      - `objective: str = 'mse'`
      - `l1_readout: float = 0.0`
      - `l1_params: float = 0.0`
      - `val_ratio: float = 0.15`
      - `early_stop_patience: int = 25`
      - `early_stop_min_delta: float = 1e-06`
      - `random_seed: int = 42`
      - `device: str = 'auto'`
      - `ood_z_threshold: float = 4.0`
      - `epsilon: float = 1e-06`
      - `verbose: bool = False`
  - `class SymbolicTorchSurrogateTrainer(BaseSurrogateTrainer)`
    - methods:
      - `__init__(self, config: SymbolicTorchTrainerConfig | None = None, *, pipeline: BasePipeline | None = None, biases: Sequence[BaseTrainingBias] | None = None, numericizer: BaseNumericizer | None = None, modality_encoders: Mapping[str, ModalityEncoder] | None = None, target_codecs: Mapping[str, TargetCodec] | None = None, target_codec: str | None = None, categorical_unknown: str = 'error') -> None`
      - `capabilities(self) -> dict[str, object]`
      - `_resolve_device(self) -> torch.device`
      - `_split_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]`
      - `_build_genome(self, X_basis: np.ndarray, Y_basis: np.ndarray)`
      - `fit(self, data: ProcessedDataset | SampleDataset) -> SymbolicSurrogateArtifact`
- Top-level functions: 0

#### Module `core.trainers.torch_trainer`
- File: `core\trainers\torch_trainer.py`
- Top-level classes: 2
  - `class TorchMLPTrainerConfig`
    - fields:
      - `artifact_id: str = 'torch_mlp_surrogate_v1'`
      - `hidden_dims: Sequence[int] = (128, 64)`
      - `activation: str = 'relu'`
      - `dropout: float = 0.0`
      - `epochs: int = 120`
      - `batch_size: int = 64`
      - `batch_shuffle: bool = True`
      - `batch_drop_last: bool = False`
      - `batch_num_workers: int = 0`
      - `batch_pin_memory: bool = False`
      - `lr: float = 0.001`
      - `weight_decay: float = 0.0001`
      - `optimizer: str = 'adamw'`
      - `optimizer_params: Mapping[str, Any] = field(default_factory=dict)`
      - `objective: str = 'mse'`
      - `val_ratio: float = 0.15`
      - `early_stop_patience: int = 20`
      - `early_stop_min_delta: float = 1e-06`
      - `random_seed: int = 42`
      - `device: str = 'auto'`
      - `ood_z_threshold: float = 4.0`
      - `verbose: bool = False`
  - `class TorchMLPSurrogateTrainer(BaseSurrogateTrainer)`
    - methods:
      - `__init__(self, config: TorchMLPTrainerConfig | None = None, *, pipeline: BasePipeline | None = None, biases: Sequence[BaseTrainingBias] | None = None, numericizer: BaseNumericizer | None = None, modality_encoders: Mapping[str, ModalityEncoder] | None = None, target_codecs: Mapping[str, TargetCodec] | None = None, target_codec: str | None = None, categorical_unknown: str = 'error') -> None`
      - `capabilities(self) -> dict[str, object]`
      - `_resolve_device(self) -> torch.device`
      - `_split_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]`
      - `fit(self, data: ProcessedDataset | SampleDataset) -> TorchMLPSurrogateArtifact`
- Top-level functions: 0

#### Module `core.trainers.trainer`
- File: `core\trainers\trainer.py`
- Top-level classes: 2
  - `class RidgeTrainerConfig`
    - fields:
      - `l2: float = 1.0`
      - `ood_z_threshold: float = 4.0`
      - `artifact_id: str = 'ridge_surrogate_v1'`
  - `class RidgeSurrogateTrainer(BaseSurrogateTrainer)`
    - methods:
      - `__init__(self, config: RidgeTrainerConfig | None = None, *, pipeline: BasePipeline | None = None, biases: Sequence[BaseTrainingBias] | None = None, numericizer: BaseNumericizer | None = None, modality_encoders: Mapping[str, ModalityEncoder] | None = None, target_codecs: Mapping[str, TargetCodec] | None = None, target_codec: str | None = None, categorical_unknown: str = 'error') -> None`
      - `_normalize_data(self, data: ProcessedDataset | SampleDataset) -> ProcessedDataset`
      - `capabilities(self) -> dict[str, object]`
      - `fit(self, data: ProcessedDataset | SampleDataset) -> LinearSurrogateArtifact`
- Top-level functions: 0

#### Module `core.trainers.xgboost_trainer`
- File: `core\trainers\xgboost_trainer.py`
- Top-level classes: 2
  - `class XGBoostTrainerConfig`
    - fields:
      - `artifact_id: str = 'xgboost_surrogate_v1'`
      - `n_estimators: int = 400`
      - `max_depth: int = 6`
      - `learning_rate: float = 0.05`
      - `subsample: float = 0.9`
      - `colsample_bytree: float = 0.9`
      - `min_child_weight: float = 1.0`
      - `gamma: float = 0.0`
      - `reg_lambda: float = 1.0`
      - `reg_alpha: float = 0.0`
      - `objective: str = 'reg:squarederror'`
      - `tree_method: str = 'hist'`
      - `n_jobs: int = -1`
      - `random_seed: int = 42`
      - `verbosity: int = 0`
      - `ood_z_threshold: float = 4.0`
  - `class XGBoostSurrogateTrainer(BaseSurrogateTrainer)`
    - methods:
      - `__init__(self, config: XGBoostTrainerConfig | None = None, *, pipeline: BasePipeline | None = None, biases: Sequence[BaseTrainingBias] | None = None, numericizer: BaseNumericizer | None = None, modality_encoders: Mapping[str, ModalityEncoder] | None = None, target_codecs: Mapping[str, TargetCodec] | None = None, target_codec: str | None = None, categorical_unknown: str = 'error') -> None`
      - `capabilities(self) -> dict[str, object]`
      - `_make_base_model(self) -> XGBRegressor`
      - `fit(self, data: ProcessedDataset | SampleDataset) -> XGBoostSurrogateArtifact`
- Top-level functions: 0

### Package `examples`

#### Module `examples.compare_trainers`
- File: `examples\compare_trainers.py`
- Top-level classes: 0
- Top-level functions: 4
  - `_build_samples(n: int = 1600, seed: int = 42) -> list[Sample]`
  - `_split_samples(samples: list[Sample], ratio: float = 0.8, seed: int = 42) -> tuple[list[Sample], list[Sample]]`
  - `_metric_report(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]`
  - `main() -> None`

#### Module `examples.init_project_scaffold`
- File: `examples\init_project_scaffold.py`
- Top-level classes: 0
- Top-level functions: 1
  - `main() -> None`

#### Module `examples.run_framework_capability_test`
- File: `examples\run_framework_capability_test.py`
- Top-level classes: 0
- Top-level functions: 6
  - `_jsonable(v: Any) -> Any`
  - `_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`
  - `_build_problem(*, n_total: int = 2200, train_ratio: float = 0.8, noise_std: float = 0.1, seed: int = 7) -> tuple[ProcessedDataset, np.ndarray, np.ndarray]`
  - `_fit(trainer_key: str, trainer_params: dict[str, Any], train_ds: ProcessedDataset, X_test: np.ndarray, y_test: np.ndarray, *, tag: str, out_root: Path) -> dict[str, Any]`
  - `_summarize_trace(run: Mapping[str, Any]) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_interval_pareto_nsgablack`
- File: `examples\run_interval_pareto_nsgablack.py`
- Top-level classes: 1
  - `class MLBlackIntervalOuterProblem`
- Top-level functions: 10
  - `_to_jsonable(v: Any) -> Any`
  - `_log10_interp(z: float, low: float, high: float) -> float`
  - `_interval_metrics(artifact: Any, X_test: np.ndarray | None, y_test: np.ndarray | None) -> tuple[float, float]`
  - `_vector_key(x: np.ndarray, ndigits: int = 8) -> str`
  - `build_problem_class(BlackBoxProblem)`
  - `_pareto_from_result(problem: Any, result: dict[str, Any]) -> list[dict[str, Any]]`
  - `_pick_compromise(pareto_rows: list[dict[str, Any]]) -> dict[str, Any] | None`
  - `_write_json(path: Path, payload: Any) -> None`
  - `_write_csv(path: Path, rows: list[dict[str, Any]]) -> None`
  - `main() -> None`

#### Module `examples.run_new_problem_advanced_benchmark`
- File: `examples\run_new_problem_advanced_benchmark.py`
- Top-level classes: 0
- Top-level functions: 5
  - `_jsonable(v: Any) -> Any`
  - `_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`
  - `_fit(trainer_key: str, trainer_params: dict[str, Any], train_ds: ProcessedDataset, X_test: np.ndarray, y_test: np.ndarray, *, tag: str, out_root: Path) -> dict[str, Any]`
  - `_trace_summary(run: Mapping[str, Any]) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_new_problem_regime_lag_test`
- File: `examples\run_new_problem_regime_lag_test.py`
- Top-level classes: 0
- Top-level functions: 6
  - `_jsonable(v: Any) -> Any`
  - `_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`
  - `_build_problem(*, n_total: int = 2800, train_ratio: float = 0.8, noise_std: float = 0.08, shift_ratio: float = 0.65, seed: int = 17) -> tuple[ProcessedDataset, np.ndarray, np.ndarray]`
  - `_fit(trainer_key: str, trainer_params: dict[str, Any], train_ds: ProcessedDataset, X_test: np.ndarray, y_test: np.ndarray, *, tag: str, out_root: Path) -> dict[str, Any]`
  - `_trace_summary(run: Mapping[str, Any]) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_nsgablack_gate_bridge_demo`
- File: `examples\run_nsgablack_gate_bridge_demo.py`
- Top-level classes: 2
  - `class GateChoice`
    - fields:
      - `feature_index: int`
      - `feature_name: str`
      - `quantile: float`
      - `threshold: float`
  - `class GateThresholdProblem(BlackBoxProblem)`
    - methods:
      - `__init__(self, *, X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_names: tuple[str, ...], gate_feature_indices: tuple[int, ...], eval_trainer_key: str = 'symbolic_stagewise', eval_trainer_params: Mapping[str, Any] | None = None, min_leaf: int = 120) -> None`
      - `_eval_gate(self, choice: GateChoice) -> tuple[np.ndarray, dict[str, Any]]`
      - `evaluate(self, x)`
      - `cache_snapshot(self) -> dict[str, Any]`
- Top-level functions: 6
  - `_jsonable(v: Any) -> Any`
  - `_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`
  - `_fit_artifact(*, trainer_key: str, trainer_params: Mapping[str, Any], X: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...], target_name: str = 'target') -> SurrogateArtifact`
  - `_decode_gate(x: np.ndarray, *, gate_feature_indices: tuple[int, ...], feature_names: tuple[str, ...], X_ref: np.ndarray, q_min: float = 0.15, q_max: float = 0.85) -> GateChoice`
  - `_fit_global_baselines(*, train_ds: ProcessedDataset, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_nsgablack_gate_bridge_v2_demo`
- File: `examples\run_nsgablack_gate_bridge_v2_demo.py`
- Top-level classes: 2
  - `class GateTreeChoice`
    - fields:
      - `root_feature_index: int`
      - `root_feature_name: str`
      - `root_quantile: float`
      - `root_threshold: float`
      - `root_temp: float`
      - `right_feature_index: int`
      - `right_feature_name: str`
      - `right_quantile: float`
      - `right_threshold: float`
      - `right_temp: float`
  - `class GateTreeV2Problem(BlackBoxProblem)`
    - methods:
      - `__init__(self, *, X_fit: np.ndarray, y_fit: np.ndarray, feature_names: tuple[str, ...], gate_feature_indices: tuple[int, ...], eval_trainer_key: str, eval_trainer_params: Mapping[str, Any], root_feature_indices: tuple[int, ...] | None = None, child_feature_indices: tuple[int, ...] | None = None, rolling_folds: int = 2, rolling_val_ratio: float = 0.18, min_leaf: int = 120) -> None`
      - `_eval_choice(self, choice: GateTreeChoice) -> tuple[np.ndarray, dict[str, Any]]`
      - `evaluate(self, x)`
      - `cache_snapshot(self) -> dict[str, Any]`
- Top-level functions: 11
  - `_jsonable(v: Any) -> Any`
  - `_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`
  - `_fit_artifact(*, trainer_key: str, trainer_params: Mapping[str, Any], X: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...], target_name: str = 'target')`
  - `_sigmoid(z: np.ndarray) -> np.ndarray`
  - `_soft_gate(x_col: np.ndarray, thr: float, temp: float) -> np.ndarray`
  - `_quantile_and_temp(x_col: np.ndarray, q: float, tau_q: float) -> tuple[float, float]`
  - `_decode_choice(x: np.ndarray, *, gate_feature_indices: tuple[int, ...] = (), root_feature_indices: tuple[int, ...] | None = None, child_feature_indices: tuple[int, ...] | None = None, feature_names: tuple[str, ...], X_ref: np.ndarray) -> GateTreeChoice`
  - `_fit_tree_predict(*, choice: GateTreeChoice, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, trainer_key: str, trainer_params: Mapping[str, Any], feature_names: tuple[str, ...], min_leaf: int) -> tuple[np.ndarray, dict[str, Any]]`
  - `_rolling_splits(n: int, *, folds: int, val_ratio: float, min_train: int) -> list[tuple[np.ndarray, np.ndarray]]`
  - `_fit_global_baselines(*, train_ds: ProcessedDataset, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_nsgablack_gate_bridge_v2_work_ci_demo`
- File: `examples\run_nsgablack_gate_bridge_v2_work_ci_demo.py`
- Top-level classes: 0
- Top-level functions: 2
  - `_build_work_dataset(*, csv_path: str, target_col: str, test_fold_col: str) -> tuple[ProcessedDataset, np.ndarray, np.ndarray]`
  - `main() -> None`

#### Module `examples.run_project_scaffold`
- File: `examples\run_project_scaffold.py`
- Top-level classes: 0
- Top-level functions: 1
  - `main() -> None`

#### Module `examples.run_stagewise_known_relation_demo`
- File: `examples\run_stagewise_known_relation_demo.py`
- Top-level classes: 0
- Top-level functions: 5
  - `_jsonable(v: Any) -> Any`
  - `_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`
  - `_build_known_relation_dataset(*, n_total: int = 3000, train_ratio: float = 0.8, noise_std: float = 0.15, seed: int = 42) -> tuple[ProcessedDataset, np.ndarray, np.ndarray, dict[str, Any]]`
  - `_fit_one(trainer_key: str, trainer_params: dict[str, Any], train_ds: ProcessedDataset, X_test: np.ndarray, y_test: np.ndarray, out_dir: Path, *, run_tag: str | None = None) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_structure_search_validation`
- File: `examples\run_structure_search_validation.py`
- Top-level classes: 0
- Top-level functions: 4
  - `_jsonable(v: Any) -> Any`
  - `_run_variant(base: ScaffoldSpec, *, trainer_key: str, trainer_params: dict[str, Any], output_dir: Path, run_name: str) -> dict[str, Any]`
  - `_safe_run_variant(*args: Any, **kwargs: Any) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_train_flow`
- File: `examples\run_train_flow.py`
- Top-level classes: 0
- Top-level functions: 2
  - `_make_dataset(n: int = 1200, seed: int = 7) -> tuple[SampleDataset, SampleDataset]`
  - `main() -> None`

#### Module `examples.run_work_ci_fixed_holiday_piecewise_demo`
- File: `examples\run_work_ci_fixed_holiday_piecewise_demo.py`
- Top-level classes: 1
  - `class PiecewiseSpec`
    - fields:
      - `gate_features: tuple[str, ...]`
      - `param_features: tuple[str, ...]`
      - `min_leaf: int`
      - `merge_rare_holiday_regimes: bool`
      - `blend_with_global: bool`
      - `blend_kappa: float`
      - `local_search_topk_features: int`
      - `local_search_max_added_terms: int`
      - `local_search_max_pair_terms: int`
      - `local_search_max_candidates_per_iter: int`
      - `local_search_candidate_keep_top: int`
      - `local_search_unary_ops: tuple[str, ...]`
      - `local_search_nested_unary_patterns: tuple[str, ...]`
      - `local_search_max_arity: int`
      - `local_search_max_expr_depth: int`
      - `local_search_overfit_guard_enabled: bool`
      - `local_search_overfit_guard_val_ratio: float`
      - `local_search_overfit_guard_min_val_samples: int`
      - `local_search_overfit_guard_random_seed: int`
      - `local_search_overfit_guard_min_val_rmse_gain: float`
      - `local_search_overfit_guard_max_gap_increase: float`
      - `local_search_overfit_guard_patience: int`
      - `local_search_overfit_guard_snapshot_min_improve: float`
      - `local_search_overfit_guard_tabu_rounds: int`
      - `local_search_overfit_guard_replace_topk: int`
      - `local_search_overfit_guard_replace_drop_topk: int`
      - `local_search_enable_grad_residual_projection: bool`
      - `local_grad_projection_topk_focus: int`
      - `local_grad_projection_partner_pool: int`
      - `local_grad_projection_topk_partners: int`
      - `local_grad_projection_topk_unary: int`
      - `local_grad_projection_partner_orders: tuple[int, ...]`
      - `local_grad_projection_enable_pair_dictionary: bool`
      - `local_grad_projection_min_abs_corr: float`
      - `local_grad_projection_max_generated: int`
- Top-level functions: 13
  - `_jsonable(v: Any) -> Any`
  - `_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`
  - `_col_index(feature_names: tuple[str, ...], cols: tuple[str, ...]) -> tuple[int, ...]`
  - `_slice_cols(X: np.ndarray, idx: tuple[int, ...]) -> np.ndarray`
  - `_fit_artifact(*, trainer_key: str, trainer_params: Mapping[str, Any], X: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...])`
  - `_parse_csv_list(raw: str) -> tuple[str, ...]`
  - `_parse_csv_int_list(raw: str, *, min_value: int = 1) -> tuple[int, ...]`
  - `_gate_key(mat: np.ndarray) -> tuple[tuple[int, ...], ...]`
  - `_hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int`
  - `_build_regime_index(keys: tuple[tuple[int, ...], ...]) -> dict[tuple[int, ...], np.ndarray]`
  - `_select_training_indices_for_regime(*, target_key: tuple[int, ...], regime_index: dict[tuple[int, ...], np.ndarray], min_leaf: int) -> tuple[np.ndarray, dict[str, Any]]`
  - `_load_work(*, csv_path: str, target_col: str, test_fold_col: str) -> ProcessedDataset`
  - `main() -> None`

#### Module `examples.run_work_ci_fixed_holiday_rolling_eval`
- File: `examples\run_work_ci_fixed_holiday_rolling_eval.py`
- Top-level classes: 1
  - `class RollingSpec`
    - fields:
      - `gate_features: tuple[str, ...]`
      - `param_features: tuple[str, ...]`
      - `min_leaf: int`
      - `merge_rare_holiday_regimes: bool`
      - `blend_with_global: bool`
      - `blend_kappa: float`
      - `local_search_topk_features: int`
      - `local_search_max_added_terms: int`
      - `local_search_max_pair_terms: int`
      - `local_search_max_candidates_per_iter: int`
      - `local_search_candidate_keep_top: int`
      - `local_search_unary_ops: tuple[str, ...]`
      - `local_search_nested_unary_patterns: tuple[str, ...]`
- Top-level functions: 9
  - `_jsonable(v: Any) -> Any`
  - `_parse_csv_list(raw: str) -> tuple[str, ...]`
  - `_safe_mean(values: list[float]) -> float`
  - `_safe_std(values: list[float]) -> float`
  - `_safe_median(values: list[float]) -> float`
  - `_load_table(*, csv_path: str, target_col: str, date_col: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]`
  - `_build_rolling_splits(*, n_samples: int, min_train_size: int, test_size: int, step_size: int, split_mode: str, train_window_size: int | None) -> list[dict[str, int]]`
  - `_evaluate_one_split(*, split_tag: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, feature_names: tuple[str, ...], spec: RollingSpec) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_work_ci_robustness_report`
- File: `examples\run_work_ci_robustness_report.py`
- Top-level classes: 2
  - `class ModelSpec`
    - fields:
      - `gate_features: tuple[str, ...]`
      - `param_features: tuple[str, ...]`
      - `min_leaf: int`
      - `merge_rare_holiday_regimes: bool`
      - `blend_with_global: bool`
      - `blend_kappa: float`
      - `local_search_topk_features: int`
      - `local_search_max_added_terms: int`
      - `local_search_max_pair_terms: int`
      - `local_search_max_candidates_per_iter: int`
      - `local_search_candidate_keep_top: int`
      - `local_search_unary_ops: tuple[str, ...]`
      - `local_search_nested_unary_patterns: tuple[str, ...]`
  - `class PiecewiseBlendedModel`
    - fields:
      - `gate_idx: tuple[int, ...]`
      - `param_idx: tuple[int, ...]`
      - `global_artifact: Any`
      - `local_models: dict[tuple[int, ...], Any]`
      - `local_effective_samples: dict[tuple[int, ...], int]`
      - `blend_kappa: float`
      - `blend_with_global: bool`
      - `training_detail: dict[str, Any]`
    - methods:
      - `predict(self, X: np.ndarray) -> np.ndarray`
- Top-level functions: 12
  - `_artifact_id(prefix: str, tag: str) -> str`
  - `_jsonable(v: Any) -> Any`
  - `_parse_csv_list(raw: str) -> tuple[str, ...]`
  - `_parse_float_list(raw: str) -> tuple[float, ...]`
  - `_load_fold_split(*, csv_path: str, target_col: str, test_fold_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]`
  - `_load_full_table(*, csv_path: str, target_col: str, date_col: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]`
  - `_build_model(*, X_train: np.ndarray, y_train: np.ndarray, feature_names: tuple[str, ...], spec: ModelSpec, tag: str) -> PiecewiseBlendedModel`
  - `_compute_param_stats(*, X_train: np.ndarray, param_idx: tuple[int, ...]) -> dict[str, np.ndarray]`
  - `_evaluate_noise_stability(*, model: PiecewiseBlendedModel, X_test: np.ndarray, y_test: np.ndarray, param_stats: dict[str, np.ndarray], noise_levels: tuple[float, ...], repeats: int, random_seed: int, clean_rmse: float) -> dict[str, Any]`
  - `_evaluate_missing_tolerance(*, model: PiecewiseBlendedModel, X_test: np.ndarray, y_test: np.ndarray, param_stats: dict[str, np.ndarray], missing_rates: tuple[float, ...], repeats: int, random_seed: int, clean_rmse: float) -> dict[str, Any]`
  - `_evaluate_drift_resistance(*, csv_path: str, target_col: str, date_col: str, spec: ModelSpec, drift_train_size: int, drift_window_size: int, drift_step_size: int) -> dict[str, Any]`
  - `main() -> None`

#### Module `examples.run_work_ci_small_data_mode_report`
- File: `examples\run_work_ci_small_data_mode_report.py`
- Top-level classes: 0
- Top-level functions: 4
  - `_parse_float_list(raw: str) -> tuple[float, ...]`
  - `_run_and_parse_summary(cmd: list[str], cwd: Path) -> tuple[dict[str, Any], str]`
  - `_bootstrap_median_ci(values: np.ndarray, *, iters: int, seed: int) -> dict[str, float]`
  - `main() -> None`

#### Module `examples.run_work_ci_train_flow`
- File: `examples\run_work_ci_train_flow.py`
- Top-level classes: 0
- Top-level functions: 1
  - `main() -> None`

#### Module `examples.work_ci_reader`
- File: `examples\work_ci_reader.py`
- Top-level classes: 1
  - `class WorkCiIntervalReader`
    - fields:
      - `csv_path: str = 'C:\\Users\\hp\\Desktop\\work\\final_pipeline_package_20260402\\04_interval_dataset\\ci_interval_opt_table.csv'`
      - `target_col: str = 'ci'`
      - `date_col: str = 'date'`
      - `test_fold_col: str = 'test_fold_10'`
      - `extra_drop_cols: Sequence[str] = ()`
    - methods:
      - `read(self) -> TrainDataBundle`
- Top-level functions: 0

### Package `numericizer`

#### Module `numericizer.__init__`
- File: `numericizer\__init__.py`
- `__all__`: `BaseNumericizer`, `DefaultNumericizer`, `ModalityEncoder`, `NumericizationPlan`, `BaseTargetCodec`, `TargetCodec`, `TargetCodecError`, `NumericTargetCodec`, `BinaryTargetCodec`, `CategoricalTargetCodec`
- Top-level classes: 0
- Top-level functions: 0

#### Module `numericizer.base`
- File: `numericizer\base.py`
- Top-level classes: 1
  - `class BaseNumericizer(ABC)`
    - methods:
      - `from_sample_dataset(self, data: SampleDataset) -> ProcessedDataset`
      - `fit(self, data: SampleDataset) -> 'BaseNumericizer'`
      - `transform_features(self, samples: Sequence[Sample]) -> np.ndarray`
      - `transform_targets(self, samples: Sequence[Sample]) -> np.ndarray`
      - `to_processed(self, data: ProcessedDataset | SampleDataset) -> ProcessedDataset`
- Top-level functions: 0

#### Module `numericizer.default`
- File: `numericizer\default.py`
- Top-level classes: 1
  - `class DefaultNumericizer(BaseNumericizer)`
    - methods:
      - `__init__(self, *, modality_encoders: Mapping[str, ModalityEncoder] | None = None, target_codecs: Mapping[str, TargetCodec] | None = None, target_codec: str | None = None, categorical_unknown: str = 'error') -> None`
      - `plan(self) -> NumericizationPlan | None`
      - `_require_fitted(self) -> tuple[NumericizationPlan, BaseTargetCodec]`
      - `_use_builtin_categorical(self, modality: str) -> bool`
      - `_encode_cell(self, cell: Cell) -> np.ndarray`
      - `_resolve_target_codec(self, raw_targets: list[Any], data: SampleDataset) -> tuple[BaseTargetCodec, str]`
      - `_select_feature_keys(self, data: SampleDataset, samples: Sequence[Sample]) -> tuple[str, ...]`
      - `fit(self, data: SampleDataset) -> 'DefaultNumericizer'`
      - `transform_features(self, samples: Sequence[Sample]) -> np.ndarray`
      - `transform_targets(self, samples: Sequence[Sample]) -> np.ndarray`
      - `encode_features_only(self, samples: Sequence[Sample]) -> np.ndarray`
      - `from_sample_dataset(self, data: SampleDataset) -> ProcessedDataset`
- Top-level functions: 11
  - `_normalize_modality(modality: str | None) -> str`
  - `_is_categorical_modality(modality: str) -> bool`
  - `_to_hashable_scalar(value: Any, *, key: str) -> Any`
  - `_build_one_hot_state(values: Sequence[Any], *, key: str, unknown: str = 'error') -> dict[str, Any]`
  - `_encode_one_hot(value: Any, *, state: Mapping[str, Any], key: str, sample_id: str) -> np.ndarray`
  - `_encode_numeric_payload(payload: Any) -> np.ndarray`
  - `_encode_scalar_payload(payload: Any) -> np.ndarray`
  - `_build_modality_encoders(custom: Mapping[str, ModalityEncoder] | None) -> dict[str, ModalityEncoder]`
  - `_build_target_codecs(custom: Mapping[str, BaseTargetCodec] | None) -> dict[str, BaseTargetCodec]`
  - `_expand_names(base: str, size: int) -> tuple[str, ...]`
  - `_extract_target_raw(sample: Sample, target_key: str) -> tuple[Any, str]`

#### Module `numericizer.plan`
- File: `numericizer\plan.py`
- Top-level classes: 1
  - `class NumericizationPlan`
    - fields:
      - `feature_keys: tuple[str, ...]`
      - `feature_sizes: Mapping[str, int]`
      - `feature_names: tuple[str, ...]`
      - `feature_modalities: Mapping[str, str]`
      - `feature_states: Mapping[str, Mapping[str, Any]]`
      - `target_key: str`
      - `target_names: tuple[str, ...]`
      - `target_codec_key: str`
      - `target_codec_state: Mapping[str, Any]`
    - methods:
      - `to_metadata(self) -> dict[str, Any]`
- Top-level functions: 0

#### Module `numericizer.target_codec`
- File: `numericizer\target_codec.py`
- Top-level classes: 5
  - `class TargetCodecError(ValueError)`
  - `class BaseTargetCodec(ABC)`
    - methods:
      - `__init__(self) -> None`
      - `fit(self, values: Sequence[Any], *, target_key: str, target_names: Sequence[str] | None = None) -> 'BaseTargetCodec'`
      - `encode(self, value: Any) -> np.ndarray`
      - `decode(self, value: np.ndarray) -> Any`
      - `output_dim(self) -> int`
      - `target_names(self) -> tuple[str, ...] | None`
      - `metadata(self) -> Dict[str, Any]`
  - `class NumericTargetCodec(BaseTargetCodec)`
    - methods:
      - `fit(self, values: Sequence[Any], *, target_key: str, target_names: Sequence[str] | None = None) -> 'NumericTargetCodec'`
      - `encode(self, value: Any) -> np.ndarray`
      - `decode(self, value: np.ndarray) -> Any`
  - `class BinaryTargetCodec(BaseTargetCodec)`
    - methods:
      - `fit(self, values: Sequence[Any], *, target_key: str, target_names: Sequence[str] | None = None) -> 'BinaryTargetCodec'`
      - `_to_binary(self, value: Any) -> int`
      - `encode(self, value: Any) -> np.ndarray`
      - `decode(self, value: np.ndarray) -> Any`
  - `class CategoricalTargetCodec(BaseTargetCodec)`
    - methods:
      - `__init__(self, *, vocab: Sequence[Any] | None = None) -> None`
      - `fit(self, values: Sequence[Any], *, target_key: str, target_names: Sequence[str] | None = None) -> 'CategoricalTargetCodec'`
      - `encode(self, value: Any) -> np.ndarray`
      - `decode(self, value: np.ndarray) -> Any`
      - `metadata(self) -> Dict[str, Any]`
- Top-level functions: 4
  - `_as_numeric_vector(value: Any) -> np.ndarray`
  - `clone_target_codec(codec: BaseTargetCodec) -> BaseTargetCodec`
  - `default_target_codecs() -> Dict[str, BaseTargetCodec]`
  - `infer_target_codec_key(values: Sequence[Any]) -> str`

### Package `pipeline`

#### Module `pipeline.__init__`
- File: `pipeline\__init__.py`
- `__all__`: `BasePipeline`, `IdentityPipeline`, `ZScorePipeline`, `create_pipeline`
- Top-level classes: 0
- Top-level functions: 1
  - `create_pipeline(name: str, state: Dict[str, Any] | None = None) -> BasePipeline`

#### Module `pipeline.base`
- File: `pipeline\base.py`
- Top-level classes: 1
  - `class BasePipeline(ABC)`
    - methods:
      - `fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'BasePipeline'`
      - `transform(self, X: np.ndarray) -> np.ndarray`
      - `fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray`
      - `state_dict(self) -> Dict[str, Any]`
      - `load_state_dict(self, state: Dict[str, Any]) -> 'BasePipeline'`
- Top-level functions: 0

#### Module `pipeline.identity`
- File: `pipeline\identity.py`
- Top-level classes: 1
  - `class IdentityPipeline(BasePipeline)`
    - methods:
      - `transform(self, X: np.ndarray) -> np.ndarray`
- Top-level functions: 1
  - `_as_2d(arr: np.ndarray) -> np.ndarray`

#### Module `pipeline.zscore`
- File: `pipeline\zscore.py`
- Top-level classes: 1
  - `class ZScorePipeline(BasePipeline)`
    - methods:
      - `__init__(self, eps: float = 1e-08) -> None`
      - `fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'ZScorePipeline'`
      - `transform(self, X: np.ndarray) -> np.ndarray`
      - `state_dict(self) -> Dict[str, Any]`
      - `load_state_dict(self, state: Dict[str, Any]) -> 'ZScorePipeline'`
- Top-level functions: 1
  - `_as_2d(arr: np.ndarray) -> np.ndarray`

### Package `project`

#### Module `project.__init__`
- File: `project\__init__.py`
- `__all__`: `TableDataSpec`, `TrainStageSpec`, `ScaffoldSpec`, `build_scaffold_spec`, `init_project`, `load_scaffold_spec`, `run_project_scaffold`
- Top-level classes: 0
- Top-level functions: 0

#### Module `project.scaffold`
- File: `project\scaffold.py`
- Top-level classes: 3
  - `class TableDataSpec`
    - fields:
      - `csv_path: str`
      - `target_col: str`
      - `date_col: str | None = 'date'`
      - `drop_cols: Sequence[str] = field(default_factory=tuple)`
      - `split_mode: str = 'fold_flag'`
      - `test_fold_col: str = 'test_fold_10'`
      - `test_ratio: float = 0.2`
      - `random_seed: int = 42`
  - `class TrainStageSpec`
    - fields:
      - `trainer_key: str = 'xgboost'`
      - `pipeline_key: str = 'identity'`
      - `trainer_params: Mapping[str, Any] = field(default_factory=dict)`
      - `pipeline_params: Mapping[str, Any] = field(default_factory=dict)`
      - `biases: Sequence[Mapping[str, Any]] = field(default_factory=tuple)`
      - `numericizer_key: str = 'default'`
      - `numericizer_params: Mapping[str, Any] = field(default_factory=dict)`
      - `eval_splits: Sequence[str] = ('train', 'test')`
      - `output_dir: str = 'runs/scaffold_run'`
      - `run_name: str = 'scaffold_run'`
  - `class ScaffoldSpec`
    - fields:
      - `data: TableDataSpec`
      - `train: TrainStageSpec`
- Top-level functions: 15
  - `_write_file(path: Path, content: str, *, overwrite: bool) -> None`
  - `_folder_readme(name: str) -> str`
  - `_root_readme(project_name: str) -> str`
  - `_start_here() -> str`
  - `_sample_train_config() -> str`
  - `_config_template() -> str`
  - `_assembly_template() -> str`
  - `_run_train_template() -> str`
  - `_prepare_data_template() -> str`
  - `init_project(target_dir: Path | str, *, force: bool = False) -> Path`
  - `_to_bias_specs(items: Sequence[Mapping[str, Any]]) -> tuple[BiasSpec, ...]`
  - `_table_to_bundle(spec: TableDataSpec) -> TrainDataBundle`
  - `build_scaffold_spec(payload: Mapping[str, Any]) -> ScaffoldSpec`
  - `load_scaffold_spec(path: str | Path) -> ScaffoldSpec`
  - `run_project_scaffold(spec: ScaffoldSpec) -> TrainFlowResult`

### Package `schema`

#### Module `schema.__init__`
- File: `schema\__init__.py`
- `__all__`: `DatasetSchema`, `FeatureSpec`, `TargetSpec`, `SchemaValidationError`, `parse_row`, `parse_rows`, `ViewBuildError`, `build_target_view`, `build_target_views`
- Top-level classes: 0
- Top-level functions: 0

#### Module `schema.parser`
- File: `schema\parser.py`
- Top-level classes: 1
  - `class SchemaValidationError(ValueError)`
- Top-level functions: 17
  - `_is_numeric_scalar(value: Any) -> bool`
  - `_validate_constraints_number(value: float, constraints: Mapping[str, Any], key: str) -> None`
  - `_validate_categorical(value: Any, *, vocab: Sequence[Any] | None, unknown: str, key: str) -> Any`
  - `_validate_numeric(value: Any, *, constraints: Mapping[str, Any], key: str) -> float`
  - `_validate_integer(value: Any, *, constraints: Mapping[str, Any], key: str) -> int`
  - `_validate_boolean(value: Any, *, key: str) -> bool`
  - `_validate_text(value: Any, *, constraints: Mapping[str, Any], key: str) -> str`
  - `_validate_sequence(value: Any, *, key: str, item_dtype: str | None, constraints: Mapping[str, Any], vocab: Sequence[Any] | None, unknown: str) -> list[Any]`
  - `_validate_matrix(value: Any, *, key: str, constraints: Mapping[str, Any]) -> np.ndarray`
  - `_validate_graph(value: Any, *, key: str) -> Mapping[str, Any]`
  - `_validate_by_dtype(value: Any, *, dtype: str, constraints: Mapping[str, Any], vocab: Sequence[Any] | None, unknown: str, item_dtype: str | None, key: str) -> Any`
  - `_validate_feature(row: Mapping[str, Any], spec: FeatureSpec) -> Cell`
  - `_validate_target(row: Mapping[str, Any], target: TargetSpec) -> Any`
  - `_target_specs(schema: DatasetSchema) -> tuple[TargetSpec, ...]`
  - `_infer_target_names(value: Any, target_key: str) -> tuple[str, ...] | None`
  - `parse_row(row: Mapping[str, Any], schema: DatasetSchema, *, sample_id: str | None = None, row_index: int | None = None) -> Sample`
  - `parse_rows(rows: Sequence[Mapping[str, Any]], schema: DatasetSchema) -> SampleDataset`

#### Module `schema.spec`
- File: `schema\spec.py`
- Top-level classes: 3
  - `class FeatureSpec`
    - fields:
      - `key: str`
      - `dtype: DType`
      - `encoder: str`
      - `required: bool = True`
      - `modality: str | None = None`
      - `vocab: Sequence[Any] | None = None`
      - `ordered: bool = False`
      - `item_dtype: DType | None = None`
      - `unknown: str = 'error'`
      - `constraints: Mapping[str, Any] = field(default_factory=dict)`
      - `meta: Mapping[str, Any] = field(default_factory=dict)`
  - `class TargetSpec`
    - fields:
      - `key: str = 'target'`
      - `dtype: DType = 'numeric'`
      - `required: bool = True`
      - `vocab: Sequence[Any] | None = None`
      - `ordered: bool = False`
      - `constraints: Mapping[str, Any] = field(default_factory=dict)`
  - `class DatasetSchema`
    - fields:
      - `features: Sequence[FeatureSpec]`
      - `targets: Sequence[TargetSpec] = field(default_factory=lambda: (TargetSpec(),))`
      - `id_key: str | None = None`
      - `strict: bool = True`
      - `description: str | None = None`
    - methods:
      - `__post_init__(self) -> None`
      - `target(self) -> TargetSpec`
      - `target_keys(self) -> tuple[str, ...]`
- Top-level functions: 0

#### Module `schema.view_builder`
- File: `schema\view_builder.py`
- Top-level classes: 1
  - `class ViewBuildError(ValueError)`
- Top-level functions: 6
  - `_sample_has_target(sample: Sample, target_key: str) -> bool`
  - `_infer_target_names(data: SampleDataset, target_key: str) -> tuple[str, ...] | None`
  - `_default_feature_keys(data: SampleDataset, *, target_key: str, exclude_target_from_features: bool) -> tuple[str, ...]`
  - `build_target_view(data: SampleDataset, target_key: str, *, feature_cell_keys: Sequence[str] | None = None, target_names: Sequence[str] | None = None, description: str | None = None, strict: bool = True, exclude_target_from_features: bool = True) -> SampleDataset`
  - `_default_target_keys(data: SampleDataset) -> tuple[str, ...]`
  - `build_target_views(data: SampleDataset, *, target_keys: Sequence[str] | None = None, feature_key_map: Mapping[str, Sequence[str]] | None = None, strict: bool = True, exclude_target_from_features: bool = True) -> Dict[str, SampleDataset]`

### Package `workflow`

#### Module `workflow.__init__`
- File: `workflow\__init__.py`
- `__all__`: `BaseDataReader`, `MemoryDataReader`, `TrainDataBundle`, `TrainFlowSpec`, `SemanticTrainFlowSpec`, `TrainFlowResult`, `run_train_flow`, `run_semantic_train_flow`
- Top-level classes: 0
- Top-level functions: 0
