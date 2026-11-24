"""mlblack context vocabulary; wire names are stable."""
from blackbase.context.context_keys import (
    KEY_CANDIDATE_REPAIRED_STATE,
    KEY_CANDIDATE_MODEL,
    KEY_CANDIDATE_UNKNOWN_STATE,
    KEY_GENERATION,
    KEY_STEP,
    KEY_PROBLEM,
    KEY_POPULATION,
    KEY_OBJECTIVES,
    KEY_INDIVIDUAL,
    KEY_BEST_X,
    KEY_BEST_OBJECTIVE,
    KEY_HISTORY,
    KEY_METADATA,
    KEY_METADATA_LAYERS,
    KEY_PROBLEM_DATA,
    KEY_CONSTRAINT_VIOLATION,
    KEY_CONSTRAINT_VIOLATIONS,
    KEY_CONSTRAINTS,
    KEY_INDIVIDUAL_ID,
    KEY_BOUNDS,
    KEY_CAPACITY,
    KEY_SHAPE,
    KEY_RUNTIME_PROJECTION_AUDIT,
    KEY_STAGE_INDEX,
    KEY_STAGE_NAME,
    KEY_STAGE_TOTAL,
    KEY_STAGE_ARTIFACTS,
    KEY_STAGE_ARTIFACT_PREFIX,
    KEY_STAGE_STATUS,
    KEY_STAGE_INPUT_ARTIFACTS,
    KEY_STAGE_OUTPUT_ARTIFACTS,
    KEY_BACKEND_CAPABILITY,
    KEY_BACKEND_CONTRACT,
    KEY_BACKEND_DEVICE,
    KEY_BACKEND_DEVICE_POLICY,
    KEY_BACKEND_NAME,
    KEY_BACKEND_REQUESTED_NAME,
    KEY_BACKEND_SESSION,
    KEY_CHECKPOINT_REF,
    KEY_PIPELINE_SLOT_CONTEXT,
    KEY_RESOURCE_AUDIT,
    KEY_RESOURCE_CONTEXT,
    KEY_RESOURCE_DEVICE,
    KEY_RESOURCE_LEASE,
    KEY_RESOURCE_CONTEXT_SHORT,
    KEY_SNAPSHOT_REF,
    KEY_STAGE_AUDIT,
    KEY_STAGE_ID,
    KEY_METRICS,
    KEY_METRICS_MC_SAMPLES,
    KEY_METRICS_MC_MEAN,
    KEY_METRICS_MC_STD,
    KEY_METRICS_MC_MIN,
    KEY_METRICS_MC_MAX,
    KEY_METRICS_SURROGATE_STD,
    KEY_METRICS_IMPLICIT_RESIDUAL,
    KEY_METRICS_IMPLICIT_ITERS,
    KEY_METRICS_IMPLICIT_SUCCESS,
    KEY_METRICS_INNER_ELAPSED_MS,
    KEY_METRICS_INNER_STATUS,
    KEY_METRICS_INNER_CALLS,
    KEY_METRICS_SOFT_ERROR_COUNT,
    KEY_METRICS_SOFT_ERROR_LAST,
    KEY_METRICS_GPU_BACKEND,
    KEY_METRICS_GPU_CALLS,
    KEY_METRICS_GPU_FALLBACKS,
    KEY_EVALUATION_COUNT,
    KEY_CONTEXT_SCHEMA,
    KEY_CONTEXT_EVENTS,
    KEY_CONTEXT_CACHE,
    KEY_DECISION_TRACE,
    KEY_CHECKPOINT_LATEST_PATH,
    KEY_CHECKPOINT_LAST_LOADED_PATH,
    KEY_SNAPSHOT_KEY,
    KEY_SNAPSHOT_BACKEND,
    KEY_SNAPSHOT_SCHEMA,
    KEY_SNAPSHOT_META,
    KEY_BEST_CANDIDATE_REF,
    KEY_POPULATION_REF,
    KEY_OBJECTIVES_REF,
    KEY_CONSTRAINT_VIOLATIONS_REF,
    KEY_HISTORY_REF,
    KEY_DECISION_TRACE_REF,
    KEY_SEQUENCE_GRAPH_REF,
    KEY_BACKEND_WARM_START_REF,
    KEY_BACKEND_SOLUTION_POOL_REF,
    KEY_BACKEND_DIAGNOSTIC_REF,
    KEY_BACKEND_CALLBACK_REF,
    CANONICAL_CONTEXT_KEYS,
    CONTEXT_KEY_SET,
    METRIC_FALLBACKS,
    METRIC_KEYS,
    normalize_context_key,
    normalize_context_keys,
    register_context_keys,
    unknown_context_keys,
    validate_context_keys,
    register_context_namespace,
    context_key_scope,
    current_context_registry,
    ContextKeyRegistry,
)

KEY_ADAPTER_BEST_STATE = "adapter.best_state"
KEY_ADAPTER_CURRENT_STATE = "adapter.current_state"
KEY_ADAPTER_SEARCH_STATE = "adapter.search_state"
KEY_AUTOGRAD_OPTIM_CONFIG = "autograd.optim.config"
KEY_ADAPTER_STATE = "adapter.state"
KEY_ARTIFACT_MODEL = "artifact.model"
KEY_ARTIFACT_REPORT = "artifact.report"
KEY_ARTIFACT_STATE = "artifact.state"
KEY_ARTIFACT_SYMBOLIC_BASIS_REF = "artifact.symbolic_basis_ref"
KEY_ARTIFACT_SYMBOLIC_TASK_REF = "artifact.symbolic_task_ref"
KEY_ARTIFACT_VIEWER = "artifact.viewer"
KEY_BASE_DECODER = "base_decoder"
KEY_BASIS_ARTIFACT_REF = "basis.artifact_ref"
KEY_BASIS_CANDIDATE_REF = "basis.candidate_ref"
KEY_BASIS_CONSENSUS = "basis.consensus"
KEY_BASIS_FITTED_REF = "basis.fitted_ref"
KEY_BASIS_METRICS = "basis.metrics"
KEY_BASIS_OVERLAP_REPORT = "basis.overlap_report"
KEY_BRANCH_SPEC = "branch.spec"
KEY_BIAS_BRANCH = "bias.branch"
KEY_BIAS_DYNAMIC_POOL = "bias.dynamic_pool"
KEY_BIAS_L2_SCALE = "bias.l2_scale"
KEY_BIAS_NOOP = "bias.noop"
KEY_BIAS_OBJECTIVE_POLICY = "bias.objective_policy"
KEY_BIAS_OBJECTIVE_WEIGHTS = "bias.objective_weights"
KEY_BIAS_SOFT_PREFERENCE = "bias.soft_preference"
KEY_BIAS_STATE_L2_PENALTY = "bias.state_l2_penalty"
KEY_BRANCH_REPRESENTATIONS = "branch_representations"
KEY_CAPABILITY_SIDE_EFFECT = "capability.side_effect"
KEY_CANDIDATE_BRANCH = "candidate.branch"
KEY_CANDIDATE_DISTRIBUTION_MODEL = "candidate.distribution_model"
KEY_CANDIDATE_INTERVAL_MODEL = "candidate.interval_model"
KEY_CANDIDATE_MODEL_SPEC = "candidate.model_spec"
KEY_CANDIDATE_OUTPUT = "candidate.output"
KEY_CANDIDATE_PROBABILITY_MODEL = "candidate.probability_model"
KEY_CANDIDATE_SYMBOLIC_BASIS_MODEL = "candidate.symbolic_basis_model"
KEY_CANDIDATE_FORECAST_MODEL = "candidate.forecast_model"
KEY_DATA = "data"
KEY_DATA_FEATURE_NAMES = "data.feature_names"
KEY_DATA_GRAPHS = "data.graphs"
KEY_DATA_IMAGE_PAIRS = "data.image_pairs"
KEY_DATA_IMAGES = "data.images"
KEY_DATA_NUMERIC_VIEW = "data.numeric_view"
KEY_DATA_PREFERENCE_PAIRS = "data.preference_pairs"
KEY_DATA_RAW_ROWS = "data.raw_rows"
KEY_DATA_SCHEMA = "data.schema"
KEY_DATA_TARGET = "data.target"
KEY_DATA_TIME_SERIES_VIEW = "data.time_series_view"
KEY_DATA_X_TRAIN = "data.X_train"
KEY_DATA_X_VALID = "data.X_valid"
KEY_DATA_Y_TRAIN = "data.y_train"
KEY_DATA_Y_VALID = "data.y_valid"
KEY_PRETRAINED_MODEL = "pretrained.model"
KEY_PRETRAINED_CHECKPOINT_MAP = "pretrained.checkpoint_map"
KEY_PRETRAINED_CHECKPOINT_REPORT = "pretrained.checkpoint_report"
KEY_PRETRAINED_TOKENIZER = "pretrained.tokenizer"
KEY_ESTIMATOR_FACTORY = "estimator.factory"
KEY_EVENT_DECISION = "event.decision"
KEY_EXPERIMENT_RECORDS = "experiment.records"
KEY_FEEDBACK_CONSTRAINTS = "feedback.constraints"
KEY_FEEDBACK_GRADIENTS = "feedback.gradients"
KEY_FEEDBACK_LOSS = "feedback.loss"
KEY_FEEDBACK_METRICS = "feedback.metrics"
KEY_FEEDBACK_OBJECTIVES = "feedback.objectives"
KEY_FEEDBACK_RESIDUALS = "feedback.residuals"
KEY_FEEDBACK_SIGNALS = "feedback.signals"
KEY_FITTED_ESTIMATOR = "fitted_estimator"
KEY_HEAD_OUTPUT = "head.output"
KEY_MODEL_PARAMETER_GRADIENT = "model.parameter_gradient"
KEY_MODEL_PREDICT = "model.predict"
KEY_MODEL_PREDICT_PARAMS = "model.predict_params"
KEY_MODEL_PREDICT_INTERVAL = "model.predict_interval"
KEY_MODEL_PREDICT_PROBA = "model.predict_proba"
KEY_MODEL_ROUTE = "model.route"
KEY_MODEL_TRANSFORM = "model.transform"
KEY_MODEL_LOGITS = "model.logits"
KEY_MODEL_HIDDEN_STATES = "model.hidden_states"
KEY_MODEL_EMBEDDINGS = "model.embeddings"
KEY_MODEL_RANKING_SCORES = "model.ranking_scores"
KEY_MODEL_PREFERENCE_SCORES = "model.preference_scores"
KEY_NEURAL_GRAPH_SPEC = "neural.graph_spec"
KEY_NEURAL_PARAMETER_LAYOUT = "neural.parameter_layout"
KEY_NEURAL_OPTIMIZER_STATE = "neural.optimizer_state"
KEY_NEURAL_TRANSFORMER_SPEC = "neural.transformer_spec"
KEY_NEURAL_HIDDEN_STATES = "neural.hidden_states"
KEY_NEURAL_ATTENTION_MAPS = "neural.attention_maps"
KEY_NEURAL_FFN_ACTIVATIONS = "neural.ffn_activations"
KEY_NEURAL_AUDIT = "neural.audit"
KEY_NEURAL_AUDIT_ATTENTION_SUMMARY = "neural.audit.attention_summary"
KEY_NEURAL_AUDIT_ATTENTION_HEAD_CORR = "neural.audit.attention_head_corr"
KEY_NEURAL_AUDIT_FFN_SUMMARY = "neural.audit.ffn_summary"
KEY_NEURAL_AUDIT_FFN_ACTIVATION_SPARSITY = "neural.audit.ffn_activation_sparsity"
KEY_ORTHOGONAL_FEATURE_MAP = "orthogonal_feature_map"
KEY_PIPELINE_FEATURE_SPACE = "pipeline.feature_space"
KEY_PIPELINE_COMPONENT_STATE = "pipeline.component_state"
KEY_PIPELINE_CONDITIONAL_FEATURES = "pipeline.conditional_features"
KEY_PIPELINE_FIT_STATE = "pipeline.fit_state"
KEY_POPULATION_CANDIDATES = "population.candidates"
KEY_POPULATION_FEEDBACK = "population.feedback"
KEY_POPULATION_SNAPSHOT_REF = "population.snapshot_ref"
KEY_PREFERENCE_REFERENCE_MODEL = "preference.reference_model"
KEY_PROBLEM_DATA_X_TRAIN = "problem.data.X_train"
KEY_PROBLEM_DATA_Y_TRAIN = "problem.data.y_train"
KEY_REPRESENTATION_NUMPY_MLP_POINT = "representation.numpy_mlp_point"
KEY_ROUTER = "router"
KEY_SIGNAL_POOL = "signal.pool"
KEY_SIGNAL_BUDGET_REMAINING_RATIO = "signal.budget.remaining_ratio"
KEY_SIGNAL_GATE_ENABLED = "signal.gate.enabled"
KEY_SYMBOLIC_ARTIFACT = "symbolic.artifact"
KEY_SYMBOLIC_ARTIFACT_SCHEMA = "symbolic.artifact_schema"
KEY_SYMBOLIC_BRANCH_REPORT = "symbolic.branch_report"
KEY_SYMBOLIC_BASIS_MODEL = "symbolic.basis_model"
KEY_SYMBOLIC_CANDIDATE_POOL = "symbolic.candidate_pool"
KEY_SYMBOLIC_CANDIDATE_LINEAGE = "symbolic.candidate_lineage"
KEY_SYMBOLIC_CANDIDATE_SCORE = "symbolic.candidate_score"
KEY_SYMBOLIC_DECODER_SPEC = "symbolic.decoder_spec"
KEY_SYMBOLIC_EQUIVALENCE_REPORT = "symbolic.equivalence_report"
KEY_SYMBOLIC_EVALUATION_EVENTS = "symbolic.evaluation_events"
KEY_SYMBOLIC_EXPRESSION_SPEC = "symbolic.expression_spec"
KEY_SYMBOLIC_FUNCTION_POOL = "symbolic.function_pool"
KEY_SYMBOLIC_FUNCTION_SPACE = "symbolic.function_space"
KEY_SYMBOLIC_FOLD_REPORT = "symbolic.fold_report"
KEY_SYMBOLIC_GENOME = "symbolic.genome"
KEY_SYMBOLIC_GRAPH_CACHE = "symbolic.graph_cache"
KEY_SYMBOLIC_GRADIENT_SIGNAL = "symbolic.gradient_signal"
KEY_SYMBOLIC_NATIVE_STRUCTURE_SCORE = "symbolic.native_structure_score"
KEY_SYMBOLIC_OVERFIT_GUARD = "symbolic.overfit_guard"
KEY_SYMBOLIC_PARAMETER_SPECS = "symbolic.parameter_specs"
KEY_SYMBOLIC_PARAMETER_VALUES = "symbolic.parameter_values"
KEY_SYMBOLIC_PATH_MEMORY = "symbolic.path_memory"
KEY_SYMBOLIC_POOL_DELTA = "symbolic.pool_delta"
KEY_SYMBOLIC_PRIMITIVE_REGISTRY = "symbolic.primitive_registry"
KEY_SYMBOLIC_REPLAY_RECORD = "symbolic.replay_record"
KEY_SYMBOLIC_SEARCH_POLICY = "symbolic.search_policy"
KEY_SYMBOLIC_SEARCH_SPACE = "symbolic.search_space"
KEY_SYMBOLIC_SIMPLIFICATION_TRACE = "symbolic.simplification_trace"
KEY_SYMBOLIC_STRUCTURE_GUARD = "symbolic.structure_guard"
KEY_SYMBOLIC_TRUTH_CONTRACT_RECOVERY = "symbolic.truth_contract_recovery"
KEY_TASK_FITTED_MODEL_REF = "task.fitted_model_ref"
KEY_TASK_METRICS = "task.metrics"
KEY_TEXT_TOKEN_IDS = "text.token_ids"
KEY_TIME_SERIES_DECOMPOSITION = "time_series.decomposition"
KEY_TIME_SERIES_HORIZON = "time_series.horizon"
KEY_TIME_SERIES_MIN_TRAIN_SIZE = "time_series.min_train_size"
KEY_TIME_SERIES_OBJECTIVE_METRICS = "time_series.objective_metrics"
KEY_TIME_SERIES_SEARCH_SPACE = "time_series.search_space"
KEY_TIME_SERIES_VALIDATION_SIZE = "time_series.validation_size"
KEY_TIME_SERIES_WINDOW_CONFIG = "time_series.window_config"
KEY_TOKENIZER_VOCAB = "tokenizer.vocab"
KEY_TRAINER_CONTEXT = "trainer.context"
KEY_TRAINER_GET_STATE = "trainer.get_state"
KEY_TRAINER_REPORT = "trainer.report"
KEY_TRAINER_SNAPSHOT_STORE = "trainer.snapshot_store"
KEY_TRAINER_STEP = "trainer.step"
KEY_TRAINING_RESULT = "training.result"
KEY_TRAINING_TASK = "training.task"

register_context_namespace("mlblack", tuple(v for k, v in globals().copy().items() if k.startswith("KEY_")))
