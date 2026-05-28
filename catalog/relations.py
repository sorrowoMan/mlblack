from __future__ import annotations

import importlib
import inspect
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .registry import CatalogEntry

_ARTIFACT_HINTS = ("artifact", "_ref", "_path", "model_path", "snapshot", "report")

_FLOW_STAGES: tuple[dict[str, Any], ...] = (
    {"id": "outer_bridge", "label": "nsgablack outer search / orchestration", "kinds": ("integration", "nsgablack_symbolic", "nsgablack_neural", "problem_bridge")},
    {"id": "resource_backend", "label": "ResourceContext / backend capability", "kinds": ("backend", "backend_capability")},
    {"id": "data_view", "label": "DataView / numericizer / feature pipeline", "kinds": ("data_view", "numericizer", "pipeline", "conditional", "symbolic_pipeline")},
    {"id": "model_space", "label": "Representation / Codec / Model spec", "kinds": ("representation", "codec", "model", "provider")},
    {"id": "output_semantics", "label": "Head / output semantics", "kinds": ("head",)},
    {"id": "problem_eval", "label": "Problem / evaluation / feedback", "kinds": ("problem",)},
    {"id": "optimization", "label": "Adapter / Bias update", "kinds": ("adapter", "bias")},
    {"id": "trainer_lifecycle", "label": "Trainer / assembly lifecycle", "kinds": ("trainer", "assembly", "core", "schema", "preset")},
    {"id": "observability", "label": "Capability / Artifact / Experiment / Catalog", "kinds": ("capability", "artifact", "experiment", "catalog", "dashboard")},
)

_KIND_USAGE: dict[str, dict[str, tuple[str, ...]]] = {
    "adapter": {
        "use_when": ("需要定义单个 inner trainer 的搜索/更新策略时使用；Adapter 只消费 feedback 和 candidate state，不直接读取业务数据。",),
        "minimal_wiring": ("Trainer lifecycle -> Adapter.propose -> Representation/Problem feedback -> Adapter.update",),
        "required_roles": ("trainer", "representation", "problem"),
    },
    "representation": {
        "use_when": ("需要定义 UnknownState 如何初始化、修复、编码或解码成 ML 语义对象时使用。",),
        "minimal_wiring": ("Adapter candidate -> Representation.repair/decode -> Model/Head -> Problem.evaluate",),
        "required_roles": ("adapter", "codec", "head", "problem"),
    },
    "codec": {
        "use_when": ("需要把优化向量、结构描述或模型 spec 转换成可评估模型对象时使用。",),
        "minimal_wiring": ("UnknownState/Spec -> Codec.layout/init/decode -> Model/Head -> Problem.evaluate",),
        "required_roles": ("representation", "problem"),
    },
    "head": {
        "use_when": ("需要定义模型输出语义时使用，例如点预测、概率、区间、piecewise 或 symbolic 输出。",),
        "minimal_wiring": ("Base decoder output -> Head -> candidate model API -> Problem metrics/loss",),
        "required_roles": ("representation", "problem"),
    },
    "problem": {
        "use_when": ("需要把数据、candidate model 和业务指标转成 Feedback/objectives/constraints/signals 时使用。",),
        "minimal_wiring": ("DataView/context + candidate model/spec -> Problem.evaluate -> Feedback",),
        "required_roles": ("representation", "adapter"),
    },
    "problem_bridge": {
        "use_when": ("需要让外部优化器通过稳定 payload 调用 mlblack inner training/evaluation 时使用。",),
        "minimal_wiring": ("Outer task payload + component_overrides + ResourceContext -> mlblack proxy -> result/artifact payload",),
        "required_roles": ("trainer", "assembly", "artifact"),
    },
    "trainer": {
        "use_when": ("需要运行一个单体 ML training/evaluation 生命周期时使用；跨 trainer 编排交给 nsgablack。",),
        "minimal_wiring": ("Trainer = Adapter + Representation + Problem + Capability + Bias",),
        "required_roles": ("adapter", "representation", "problem"),
    },
    "data_view": {
        "use_when": ("需要把原始/领域数据固定成 typed data boundary 时使用，例如 numeric supervised rows、image tensors、graph tensors、preference pairs、contrastive pairs 或 ordered time-series。",),
        "minimal_wiring": ("raw/domain data -> DataView -> Pipeline/Numericizer/Problem -> Feedback",),
        "required_roles": ("pipeline", "problem"),
    },
    "pipeline": {
        "use_when": ("需要在进入 Problem 前转换 NumericDataView、特征、target 或轻量 pipeline state 时使用。",),
        "minimal_wiring": ("NumericDataView -> PipelineComponent.fit_transform/transform -> NumericDataView -> Problem",),
        "required_roles": ("numericizer", "problem"),
    },
    "numericizer": {
        "use_when": ("需要把原始 rows/schema 编成 NumericDataView，作为监督/评估问题的稳定数据入口时使用。",),
        "minimal_wiring": ("raw rows + schema -> Numericizer -> NumericDataView -> Pipeline/Problem",),
        "required_roles": ("pipeline", "problem"),
    },
    "conditional": {
        "use_when": ("需要构造条件特征、router、gate 或 branch selection 语义时使用。",),
        "minimal_wiring": ("NumericDataView/primitive config -> conditional component -> feature/router output -> Representation/Head/Problem",),
        "required_roles": ("pipeline", "representation", "head"),
    },
    "symbolic_pipeline": {
        "use_when": ("需要生成、筛选、缓存或审计 symbolic function pool / basis / grammar 时使用。",),
        "minimal_wiring": ("Data/context + symbolic grammar/pool -> symbolic pipeline output -> symbolic Representation/Problem",),
        "required_roles": ("representation", "problem", "nsgablack_symbolic"),
    },
    "bias": {
        "use_when": ("需要表达软偏好、目标重权重或候选池倾向时使用；不要替代硬约束或 Trainer 生命周期。",),
        "minimal_wiring": ("Problem Feedback -> Bias.adjust/project -> Adapter.update",),
        "required_roles": ("problem", "adapter"),
    },
    "capability": {
        "use_when": ("需要 checkpoint、experiment tracking、resource audit、report 等生命周期副作用时使用。",),
        "minimal_wiring": ("Trainer lifecycle hook -> Capability -> state/artifact/report",),
        "required_roles": ("trainer",),
    },
    "model": {
        "use_when": ("需要表达可预测、可描述、可组合的模型语义对象时使用；训练编排不放在 model 内。",),
        "minimal_wiring": ("Representation/Artifact/Spec -> Model.predict/describe -> Problem/Artifact",),
        "required_roles": ("representation", "problem"),
    },
    "provider": {
        "use_when": ("需要把外部/统计/领域拟合能力封装成稳定 ML 语义 surface 时使用；Provider 负责 fit/build，不拥有 Trainer lifecycle 或资源调度。",),
        "minimal_wiring": ("Spec + DataView + optional backend/context -> Provider.fit/build -> Model/Artifact -> Problem",),
        "required_roles": ("data_view", "model", "problem"),
    },
    "backend": {
        "use_when": ("需要选择 numpy/torch/sklearn/xgboost 等执行能力时使用；资源授权仍来自外层 ResourceContext。",),
        "minimal_wiring": ("ResourceContext/backend.session -> backend contract -> Adapter/Codec/Problem",),
        "required_roles": ("adapter", "codec", "problem"),
    },
    "backend_capability": {
        "use_when": ("需要确认某个 backend 是否支持梯度、batch、参数布局、neural lowering、mixed precision 或 resume 边界时使用。",),
        "minimal_wiring": ("backend capability matrix -> provider/session contract check -> consuming component",),
        "required_roles": ("backend",),
    },
    "integration": {
        "use_when": ("需要把 mlblack inner training/evaluation 暴露给 nsgablack 或其它外层系统时使用。",),
        "minimal_wiring": ("nsgablack task/component_overrides/ResourceContext -> mlblack bridge -> artifact/result payload",),
        "required_roles": ("problem_bridge", "trainer", "artifact"),
    },
    "nsgablack_symbolic": {
        "use_when": ("需要让 nsgablack 外层搜索 symbolic basis/function pool/task，而 mlblack 负责内层模型语义和参数拟合时使用。",),
        "minimal_wiring": ("nsgablack outer candidate -> symbolic integration problem -> mlblack fitter/artifact -> feedback.objectives",),
        "required_roles": ("symbolic_pipeline", "problem", "artifact"),
    },
    "nsgablack_neural": {
        "use_when": ("需要让 nsgablack 搜 neural spec/结构/外层任务，而 mlblack 负责 neural codec/problem/backend 能力时使用。",),
        "minimal_wiring": ("nsgablack outer candidate/spec -> mlblack neural codec/problem -> artifact/result payload",),
        "required_roles": ("codec", "problem", "backend"),
    },
    "assembly": {
        "use_when": ("需要把一个 inner trainer 的 ML 组件按标准 spec 装起来时使用；不承担 workflow/runtime 编排。",),
        "minimal_wiring": ("TrainerAssemblySpec + component_overrides -> build_trainer -> ComposableTrainer",),
        "required_roles": ("trainer", "adapter", "representation", "problem"),
    },
    "schema": {
        "use_when": ("需要声明可序列化 scaffold/config/spec 边界时使用。",),
        "minimal_wiring": ("config/schema payload -> assembly/build_trainer -> component construction",),
        "required_roles": ("assembly",),
    },
    "preset": {
        "use_when": ("需要复用一套常用 ML 组件组合，但仍保持可被 component_overrides 替换时使用。",),
        "minimal_wiring": ("preset builder -> TrainerAssemblySpec/build_trainer -> trainer",),
        "required_roles": ("assembly", "trainer"),
    },
    "catalog": {
        "use_when": ("需要查询、物化、同步或导出框架组件知识库时使用。",),
        "minimal_wiring": ("Catalog registry/materialized DB -> query/facet/show -> dashboard/API",),
        "required_roles": ("dashboard",),
    },
    "dashboard": {
        "use_when": ("需要把 catalog、experiment、artifact 或 backend matrix 变成可读 UI/HTML/API 时使用。",),
        "minimal_wiring": ("materialized DB/artifact store -> dashboard renderer -> browser/report",),
        "required_roles": ("catalog",),
    },
    "experiment": {
        "use_when": ("需要查询训练运行记录、指标、事件或 experiment store 时使用。",),
        "minimal_wiring": ("experiment store -> query/facet/export -> report/dashboard",),
        "required_roles": ("capability", "dashboard"),
    },
    "core": {
        "use_when": ("需要理解 mlblack 主干协议、状态对象、contract 或基础抽象时使用。",),
        "minimal_wiring": ("core protocol/base class -> concrete component -> Trainer/Problem/Adapter lifecycle",),
        "required_roles": ("trainer",),
    },
}

_EXACT_USAGE_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "codec.neural_graph": {
        "use_when": (
            "已有 NeuralGraphSpec，并希望把优化器搜索到的一维参数向量 flat parameter state 解码成可调用的神经网络模型时使用。",
            "适合把神经网络结构和参数纳入 mlblack/nsgablack 统一搜索空间：外层搜参数或结构，codec 负责还原成模型。",
            "MLP route 可以本地 numpy fallback；非 MLP / 图结构 route 需要通过 backend.session 调用 neural.lowering 能力。",
        ),
        "minimal_wiring": (
            "NeuralGraphSpec + UnknownState(flat params) -> NeuralGraphCodec.parameter_layout/init_values/decode -> neural model",
            "Representation 持有 NeuralGraphCodec；Problem 调用 decoded model 计算 loss/metrics；Adapter 根据 feedback 更新 flat params。",
            "如果走 torch/jax/tensorflow lowering：ResourceContext/backend.session -> backend.decode_neural_graph -> model artifact/result payload。",
        ),
        "required_roles": ("NeuralGraphSpec", "backend capability: parameters.layout / parameters.init / neural.lowering", "representation", "problem"),
        "config_keys": ("spec", "init_scale", "random_seed", "representation_name", "backend.session"),
    },
    "pipeline.model_conditioned_target": {
        "use_when": (
            "已有 reference model，并希望把下一阶段 target 变成 residual、prediction 或追加预测特征时使用。",
            "适合残差学习、stacking、boosting-like stage；它只是数据变换组件，不是新的 mlblack workflow。",
        ),
        "minimal_wiring": (
            "NumericDataView + reference_model.predict(X) -> ModelConditionedTargetComponent -> transformed NumericDataView",
            "外层阶段/多模型编排交给 nsgablack；mlblack 只负责这一个数据变换组件。",
        ),
        "required_roles": ("data_view", "reference_model", "problem"),
        "config_keys": ("mode", "reference_name", "append_prediction_feature", "prediction_feature_name", "reference_context_key"),
    },
    "model.integrated_prediction": {
        "use_when": (
            "已有多个 fitted component models，并希望把它们的 prediction 按 additive/mean 等策略组合成一个集成预测模型时使用。",
            "它只组合预测，不拥有训练编排；多模型训练、选择和资源编排应交给 nsgablack。",
        ),
        "minimal_wiring": ("component models + PredictionIntegrationSpec + PredictionIOContract -> IntegratedPredictionModel.predict",),
        "required_roles": ("component_models", "prediction_io_contract", "problem"),
        "config_keys": ("components", "integration", "io_contract", "component_order", "weights", "intercept"),
    },
}

_PATTERN_RULES: tuple[tuple[tuple[str, ...], dict[str, tuple[str, ...]]], ...] = (
    (("gradient_descent",), {"use_when": ("当 Problem 能产出 feedback.gradients，且 UnknownState 可以用一阶梯度直接更新时使用。",), "minimal_wiring": ("feedback.gradients + candidate.unknown_state -> GradientDescentAdapter.update -> adapter.current_state/population.candidates",)}),
    (("functional_backprop",), {"use_when": ("当梯度由 Problem/Backend 的 functional route 统一产出，而不是 Adapter 私下读取模型内部细节时使用。",), "minimal_wiring": ("candidate.model + backend.contract + problem-owned gradients -> FunctionalBackpropAdapter -> optimizer step",)}),
    (("torch_backprop",), {"use_when": ("当参数向量对应 torch/MLP 训练路径，并需要 batch/device/optimizer state 时使用。",), "minimal_wiring": ("ResourceContext/device + numpy_mlp representation + training data -> TorchBackpropAdapter.step",)}),
    (("random_search",), {"use_when": ("当模型不可微、head 不提供梯度，或需要黑盒 baseline 搜索时使用。",), "minimal_wiring": ("current best + sampling scale -> candidates -> Problem feedback -> keep/improve state",)}),
    (("estimator",), {"use_when": ("当 UnknownState 解码为 sklearn/xgboost/tree 等外部 estimator spec，并由 Problem 负责 fit/score 时使用。",), "minimal_wiring": ("EstimatorSpecRepresentation -> estimator factory/problem fit -> objectives -> estimator search/update",)}),
    (("linear",), {"use_when": ("当 UnknownState 表示线性权重/截距，并需要解码成点预测模型时使用。",), "minimal_wiring": ("flat weights -> linear codec/representation -> linear model -> supervised problem",)}),
    (("numpy_mlp",), {"use_when": ("当希望用轻量 numpy MLP 表征做本地解码、预测或作为 torch backprop 的参数布局基础时使用。",), "minimal_wiring": ("flat parameters -> NumpyMLP codec/representation -> NumpyMLPPointModel.predict",)}),
    (("symbolic",), {"use_when": ("当候选空间涉及符号表达式、basis-set、function pool、grammar 或符号参数拟合时使用。",), "minimal_wiring": ("symbolic genome/spec/pool + UnknownState -> symbolic component -> symbolic model/problem/artifact",)}),
    (("probability",), {"use_when": ("当模型输出需要概率语义、predict_proba 或概率校准时使用。",), "minimal_wiring": ("base decoder logits -> probability head -> predict_proba/predict -> classification problem",)}),
    (("softmax",), {"use_when": ("当多分类模型需要每类一个 decoder block 并输出 softmax probability 时使用。",), "minimal_wiring": ("per-class decoder blocks -> SoftmaxHead -> multiclass probability model",)}),
    (("logistic",), {"use_when": ("当二分类模型需要把 scalar logit 包装成 binary predict_proba 输出时使用。",), "minimal_wiring": ("scalar logit decoder -> BinaryLogisticHead -> binary probability model",)}),
    (("interval",), {"use_when": ("当回归输出需要区间上下界、中心半径或不确定性区间语义时使用。",), "minimal_wiring": ("base decoder blocks -> interval head -> predict_interval -> interval regression problem",)}),
    (("piecewise",), {"use_when": ("当输出或模型结构需要按 router/branch 分段组合时使用。",), "minimal_wiring": ("router + branch representations/heads -> piecewise model -> piecewise problem",)}),
    (("classification",), {"use_when": ("当 candidate model 输出类别或概率，并需要 accuracy/log-loss/AUC/F1 等分类反馈时使用。",), "minimal_wiring": ("candidate probability/class model + X/y -> classification Problem.evaluate -> feedback.metrics/objectives",)}),
    (("regression",), {"use_when": ("当 candidate model 输出连续值，并需要 RMSE/MAE/R2/residual/gradient 等回归反馈时使用。",), "minimal_wiring": ("candidate model + X/y -> regression Problem.evaluate -> loss/residuals/gradients",)}),
    (("transformer",), {"use_when": ("当模型语义是 transformer/tokenizer/pretrained route，并需要把文本任务接入统一 Problem/Codec/Backend 协议时使用。",), "minimal_wiring": ("tokenizer/model spec + data -> transformer bridge/problem -> feedback/artifact",)}),
    (("cnn",), {"use_when": ("当输入是图像/张量并需要 CNN 结构或图像任务评估时使用。",), "minimal_wiring": ("image tensors + candidate CNN model -> image problem -> feedback",)}),
    (("gnn",), {"use_when": ("当输入是 graph data，并需要 GNN 模型或图分类/图表示评估时使用。",), "minimal_wiring": ("graph batch + candidate GNN model -> graph problem -> feedback",)}),
    (("contrastive",), {"use_when": ("当训练目标是 pair/embedding 对比学习而不是单点监督标签时使用。",), "minimal_wiring": ("paired data + encoder model -> contrastive objective -> feedback",)}),
    (("dpo",), {"use_when": ("当训练目标是 preference pairs / DPO 风格偏好优化时使用。",), "minimal_wiring": ("preference pairs + policy/reference scores -> DPO problem -> feedback",)}),
    (("numericizer",), {"use_when": ("当原始 rows/schema 尚未变成 NumericDataView，或需要可审计的 feature/target 编码时使用。",), "minimal_wiring": ("raw rows + schema/target codec -> NumericDataView -> pipeline/problem",)}),
    (("feature_space",), {"use_when": ("当需要记录、传播或审计特征空间元数据，而不是实际训练模型时使用。",), "minimal_wiring": ("NumericDataView -> FeatureSpaceComponent -> metadata-enriched NumericDataView",)}),
    (("conditional",), {"use_when": ("当数据流或模型输出需要 gate、branch、primitive feature 或 conditional routing 时使用。",), "minimal_wiring": ("conditional primitive/composer -> feature/router output -> representation/head/problem",)}),
    (("dynamic_pool",), {"use_when": ("当候选函数池、branch pool 或搜索池需要根据 residual/gradient/signal 动态扩张和裁剪时使用。",), "minimal_wiring": ("feedback/signal + pool policy -> updated pool hint -> outer search/representation",)}),
    (("checkpoint",), {"use_when": ("当训练过程需要可恢复 state snapshot，或需要把 trainer state 写入 artifact/snapshot store 时使用。",), "minimal_wiring": ("Trainer lifecycle -> CheckpointCapability -> TrainerStateArtifact/SnapshotStore",)}),
    (("experiment",), {"use_when": ("当需要记录 run、step、metric、event，并支持后续查询/可视化时使用。",), "minimal_wiring": ("Trainer/Capability events -> experiment store -> query/dashboard",)}),
    (("resource_audit",), {"use_when": ("当需要审计外层注入的 ResourceContext 是否真正传到 mlblack inner training 时使用。",), "minimal_wiring": ("ResourceContext -> ResourceAuditCapability -> run report/artifact metadata",)}),
    (("catalog",), {"use_when": ("当需要查询组件用途、契约字段、关联组件、运行流程或 DB-only catalog UI 时使用。",), "minimal_wiring": ("materialize_catalog_db -> catalog store -> query/show/dashboard",)}),
    (("postgres",), {"use_when": ("当 catalog 需要落到 PostgreSQL 供多进程/远端查询，而不是只用本地 SQLite 时使用。",), "minimal_wiring": ("PostgreSQL URL -> PostgresCatalogStore -> materialize/query",)}),
    (("sqlite",), {"use_when": ("当 catalog 需要本地默认 DB 快照和离线查询时使用。",), "minimal_wiring": (".mlblack/catalog.sqlite -> SQLiteCatalogStore -> webui/query",)}),
    (("backend",), {"use_when": ("当组件需要显式执行能力边界，例如 parameters.layout、neural.lowering、optimizer step 或 estimator fit 时使用。",), "minimal_wiring": ("backend provider/session -> capability contract -> consuming Adapter/Codec/Problem",)}),
    (("nsgablack",), {"use_when": ("当能力涉及外层搜索、阶段、嵌套评估、Pareto 或 ResourceContext 注入时使用；mlblack 不在内部私造 workflow/runtime。",), "minimal_wiring": ("nsgablack outer solver/task -> mlblack bridge/proxy -> Feedback/artifact/result payload",)}),
    (("time_series",), {"use_when": ("当数据具有时间顺序、lag/window/horizon、rolling backtest 或 forecast 语义时使用。",), "minimal_wiring": ("TimeSeriesDataView -> TimeSeriesWindowing/ForecastRepresentation -> forecast model -> TimeSeriesForecastingProblem",)}),
    (("forecast",), {"use_when": ("当组件负责生成、包装或评估未来 horizon 预测时使用。",), "minimal_wiring": ("history + horizon + optional exogenous_future -> forecast(...) -> metrics/artifact",)}),
    (("rolling",), {"use_when": ("当需要滚动起点 backtest，而不是一次性 holdout tail 评估时使用。",), "minimal_wiring": ("rolling origins -> repeated forecast(history, horizon) -> aggregate RMSE/MAE/MAPE/MASE",)}),
    (("baseline_forecast",), {"use_when": ("当需要在复杂模型前建立 naive/seasonal-naive/moving-average 预测基线，或让外层搜索基线参数时使用。",), "minimal_wiring": ("UnknownState(strategy/window/seasonal_period) -> BaselineForecastRepresentation.decode -> NaiveForecastModel",)}),
)

_RELATION_LABELS = {
    "context_upstream": "上游上下文字段生产者",
    "context_downstream": "下游上下文字段消费者",
    "artifact_upstream": "上游 Artifact 生产者",
    "artifact_downstream": "下游 Artifact 消费者",
    "role_companions": "同流程建议搭档",
    "companions": "显式关联条目",
    "linked_by": "反向引用",
}


def build_relation_payload_index(entries: Sequence[CatalogEntry]) -> dict[str, dict[str, Any]]:
    all_entries = tuple(entries)
    return {entry.key: build_entry_relation_payload(entry, all_entries=all_entries) for entry in all_entries}


def build_entry_relation_payload(entry: CatalogEntry, *, all_entries: Sequence[CatalogEntry]) -> dict[str, Any]:
    fields = relation_fields(entry)
    neighbors = _neighbor_payload(entry, fields=fields, all_entries=tuple(all_entries))
    return {
        "key": entry.key,
        "kind": entry.kind,
        "fields": fields,
        "usage": usage_profile(entry),
        "neighbors": neighbors,
        "field_refs": _field_reference_rows(entry, fields=fields, all_entries=tuple(all_entries)),
        "flow": flow_payload(entry),
        "relation_cards": _relation_cards(neighbors),
    }


def relation_fields(entry: CatalogEntry) -> dict[str, tuple[str, ...]]:
    contract = dict(entry.contract)
    metadata = dict(entry.metadata)
    context_requires = _values(contract.get("context_requires"))
    context_provides = _values(contract.get("context_provides"))
    context_mutates = _values(contract.get("context_mutates"))
    context_cache = _values(contract.get("context_cache"))
    artifact_requires = _unique(
        _values(contract.get("artifact_requires")),
        tuple(value for value in context_requires if _looks_like_artifact(value)),
    )
    artifact_provides = _unique(
        _values(contract.get("artifact_provides")),
        tuple(value for value in (*context_provides, *context_mutates) if _looks_like_artifact(value)),
    )
    phase_in = _unique(_values(contract.get("phase_in")), _values(metadata.get("phase_in")))
    phase_out = _unique(_values(contract.get("phase_out")), _values(metadata.get("phase_out")))
    return {
        "context_requires": context_requires,
        "context_provides": context_provides,
        "context_mutates": context_mutates,
        "context_cache": context_cache,
        "requires_metrics": _values(contract.get("requires_metrics")),
        "artifact_requires": artifact_requires,
        "artifact_provides": artifact_provides,
        "phase_in": phase_in,
        "phase_out": phase_out,
        "resource_refs": tuple(value for value in _unique(context_requires, context_provides, context_mutates) if "resource" in value.lower()),
    }


def usage_profile(entry: CatalogEntry) -> dict[str, Any]:
    metadata = dict(entry.metadata)
    contract = dict(entry.contract)
    fields = relation_fields(entry)
    base = dict(_KIND_USAGE.get(entry.kind, {}))
    specific = _specific_usage_hint(entry)
    dynamic = _dynamic_usage_hint(entry, fields=fields)
    use_when = _unique(_values(metadata.get("use_when")), specific.get("use_when", ()), dynamic.get("use_when", ()), _summary_usage(entry), base.get("use_when", ()))
    minimal_wiring = _unique(_values(metadata.get("minimal_wiring")), specific.get("minimal_wiring", ()), dynamic.get("minimal_wiring", ()), _contract_wiring(fields), base.get("minimal_wiring", ()))
    required_roles = _unique(_values(metadata.get("required_roles")), specific.get("required_roles", ()), dynamic.get("required_roles", ()), _roles_from_fields(entry, fields=fields), base.get("required_roles", ()))
    config_keys = _unique(_values(metadata.get("config_keys")), specific.get("config_keys", ()), dynamic.get("config_keys", ()), _values(contract.get("config_keys")), _signature_config_keys(entry))
    notes = _unique(dynamic.get("notes", ()), _capability_notes(entry, fields=fields))
    return {
        "use_when": use_when or ("组件用途需要结合导入路径、契约字段和源码说明确认。",),
        "minimal_wiring": minimal_wiring or (f"{entry.title} -> downstream component",),
        "required_roles": required_roles or ("trainer",),
        "config_keys": config_keys or ("无固定构造配置；主要由上下文、上游组件或默认参数提供。",),
        "notes": notes,
    }


def _specific_usage_hint(entry: CatalogEntry) -> Mapping[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {}
    if entry.key in _EXACT_USAGE_HINTS:
        merged = _merge_hint_dicts(merged, _EXACT_USAGE_HINTS[entry.key])
    key_text = _entry_text(entry)
    if "neural_graph" in key_text or "neuralgraph" in key_text:
        merged = _merge_hint_dicts(merged, _EXACT_USAGE_HINTS["codec.neural_graph"])
    for needles, hint in _PATTERN_RULES:
        if needles == ("linear",) and entry.kind not in {"representation", "codec", "model", "head"}:
            continue
        if all(needle in key_text for needle in needles):
            merged = _merge_hint_dicts(merged, hint)
    return merged


def _dynamic_usage_hint(entry: CatalogEntry, *, fields: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    title = str(entry.title or entry.key)
    kind = str(entry.kind or "component")
    import_path = str(entry.import_path or "")
    module_path, _, symbol_name = import_path.partition(":")
    architecture_path = str(dict(entry.metadata).get("architecture_path", "") or module_path.removeprefix("mlblack."))
    methods = _public_operation_methods(entry)
    use_when: list[str] = []
    minimal_wiring: list[str] = []
    notes: list[str] = []

    if architecture_path:
        use_when.append(f"架构位置：{architecture_path}；这是 `{kind}` 层的 `{title}`，不是孤立工具函数。")
    if module_path and symbol_name:
        notes.append(f"导入路径：{module_path}:{symbol_name}。")
    if methods:
        notes.append("主要操作方法：" + ", ".join(methods) + "。")

    if kind == "adapter":
        minimal_wiring.append(f"{title}.propose/update 读取 {_fmt_values(fields.get('context_requires'))}，写出 {_fmt_values(_unique(fields.get('context_provides'), fields.get('context_mutates')))}。")
    elif kind in {"representation", "codec"}:
        minimal_wiring.append(f"{title} 把 candidate/UnknownState 或 spec 转成下游 Problem 可评估的模型语义对象。")
    elif kind == "head":
        minimal_wiring.append(f"{title} 包装 base decoder 输出，暴露 Problem 需要的 predict/predict_proba/predict_interval 等语义。")
    elif kind == "problem":
        minimal_wiring.append(f"{title}.evaluate 消费 {_fmt_values(fields.get('context_requires'))}，产出 {_fmt_values(fields.get('context_provides'))}。")
    elif kind in {"data_view", "pipeline", "numericizer", "conditional", "symbolic_pipeline"}:
        minimal_wiring.append(f"{title} 位于数据/特征进入 Problem 前的准备链，输入输出以 DataView 或 context 字段为边界。")
    elif kind in {"nsgablack_symbolic", "nsgablack_neural", "integration", "problem_bridge"}:
        minimal_wiring.append(f"{title} 是跨框架边界：外层 nsgablack 负责编排/资源，mlblack 只提供 inner semantic/evaluation surface。")
    elif kind in {"capability", "dashboard", "catalog", "experiment"}:
        minimal_wiring.append(f"{title} 增强观测、查询、持久化或展示，不改变模型优化语义。")
    elif kind == "model":
        minimal_wiring.append(f"{title} 表示可预测/可描述的模型对象，由 Representation、Artifact 或组合组件产生后交给 Problem。")
    elif kind == "provider":
        minimal_wiring.append(f"{title} 根据 Spec/DataView 构建模型或 artifact；外层阶段、并行和资源授权仍由 nsgablack/Trainer context 提供。")

    contract_line = _contract_summary_line(fields)
    if contract_line:
        notes.append(contract_line)
    return {"use_when": tuple(use_when), "minimal_wiring": tuple(minimal_wiring), "notes": tuple(notes)}


def _summary_usage(entry: CatalogEntry) -> tuple[str, ...]:
    summary = str(entry.summary or "").strip()
    return (f"组件语义：{summary}",) if summary else tuple()


def _contract_wiring(fields: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    requires = _fmt_values(fields.get("context_requires"))
    provides = _fmt_values(_unique(fields.get("context_provides"), fields.get("context_mutates"), fields.get("artifact_provides")))
    if requires == "无显式字段" and provides == "无显式字段":
        return tuple()
    return (f"契约字段链：读取 {requires}；产出/修改 {provides}。",)


def _roles_from_fields(entry: CatalogEntry, *, fields: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    field_values: list[str] = []
    for value in fields.values():
        field_values.extend(_values(value))
    text = " ".join(field_values).lower()
    roles: list[str] = []
    if "unknown_state" in text or "candidate.model" in text or "candidate.model_spec" in text:
        roles.append("representation")
    if "feedback" in text or str(entry.kind) == "adapter":
        roles.append("adapter")
    if "data." in text or "numeric_view" in text:
        roles.append("data_view")
    if "backend" in text:
        roles.append("backend")
    if "resource" in text:
        roles.append("ResourceContext")
    if "artifact" in text or "snapshot" in text:
        roles.append("artifact")
    if str(entry.kind) in {"head", "codec", "representation"}:
        roles.append("problem")
    return tuple(roles)


def _capability_notes(entry: CatalogEntry, *, fields: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    contract = dict(entry.contract)
    notes: list[str] = []
    if contract.get("supports_gradient") is True:
        notes.append("支持梯度路径；应确认 Problem/Backend 的梯度协议与 Adapter 消费字段对齐。")
    elif contract.get("supports_gradient") is False:
        notes.append("不依赖梯度；通常走黑盒、外部 estimator、区间或离散搜索路径。")
    if contract.get("supports_batch") is True:
        notes.append("支持 batch / population 风格评估或更新。")
    if contract.get("supports_resume") is True:
        notes.append("支持 resume/state 恢复；需要与 snapshot/artifact 边界一起审计。")
    if fields.get("resource_refs"):
        notes.append("包含 resource 字段；运行时必须读取外层注入的 ResourceContext，不应在 mlblack 内私自分配资源。")
    if entry.kind.startswith("nsgablack") or entry.kind in {"integration", "problem_bridge"}:
        notes.append("跨框架入口：nsgablack 负责编排、并行、Pareto 和资源授权；mlblack 只暴露 inner task/result/artifact surface。")
    return tuple(notes)


def flow_payload(entry: CatalogEntry) -> dict[str, Any]:
    current_stage = _stage_for_kind(entry.kind)
    nodes = []
    for stage in _FLOW_STAGES:
        nodes.append({"id": stage["id"], "label": stage["label"], "active": stage["id"] == current_stage, "kinds": tuple(stage["kinds"])})
    edges = tuple({"source": _FLOW_STAGES[index]["id"], "target": _FLOW_STAGES[index + 1]["id"]} for index in range(len(_FLOW_STAGES) - 1))
    return {
        "current_stage": current_stage,
        "current_stage_label": next((str(node["label"]) for node in nodes if node["active"]), ""),
        "nodes": tuple(nodes),
        "edges": edges,
    }


def relation_search_text(payload: Mapping[str, Any]) -> str:
    return " ".join(_flatten_strings(payload)).lower()


def _neighbor_payload(entry: CatalogEntry, *, fields: Mapping[str, Sequence[str]], all_entries: Sequence[CatalogEntry]) -> dict[str, tuple[dict[str, Any], ...]]:
    by_key = {item.key: item for item in all_entries}
    producers = _entries_matching(all_entries, skip_key=entry.key, values=fields.get("context_requires", ()), target_fields=("context_provides", "context_mutates"))
    consumers = _entries_matching(all_entries, skip_key=entry.key, values=_unique(fields.get("context_provides", ()), fields.get("context_mutates", ())), target_fields=("context_requires",))
    artifact_producers = _entries_matching(all_entries, skip_key=entry.key, values=fields.get("artifact_requires", ()), target_fields=("artifact_provides", "context_provides", "context_mutates"), artifact_only=True)
    artifact_consumers = _entries_matching(all_entries, skip_key=entry.key, values=fields.get("artifact_provides", ()), target_fields=("artifact_requires", "context_requires"), artifact_only=True)
    role_companions = _role_companions(entry, all_entries=all_entries)
    explicit_keys = _values(entry.metadata.get("companions") if isinstance(entry.metadata, Mapping) else ())
    explicit_companions = tuple(_entry_payload(by_key[key]) for key in explicit_keys if key in by_key and key != entry.key)
    missing_companions = tuple(key for key in explicit_keys if key not in by_key)
    return {
        "context_upstream": tuple(_entry_payload(item) for item in producers),
        "context_downstream": tuple(_entry_payload(item) for item in consumers),
        "artifact_upstream": tuple(_entry_payload(item) for item in artifact_producers),
        "artifact_downstream": tuple(_entry_payload(item) for item in artifact_consumers),
        "role_companions": tuple(_entry_payload(item) for item in role_companions),
        "companions": explicit_companions,
        "missing_companions": tuple({"key": key, "title": key, "kind": "missing", "summary": ""} for key in missing_companions),
        "linked_by": tuple(_entry_payload(item) for item in _linked_by(entry, all_entries=all_entries)),
    }


def _field_reference_rows(entry: CatalogEntry, *, fields: Mapping[str, Sequence[str]], all_entries: Sequence[CatalogEntry]) -> tuple[dict[str, Any], ...]:
    values = _unique(fields.get("context_requires", ()), fields.get("context_provides", ()), fields.get("context_mutates", ()), fields.get("artifact_requires", ()), fields.get("artifact_provides", ()))
    rows = []
    for value in values:
        producers = _entries_matching(all_entries, skip_key="", values=(value,), target_fields=("context_provides", "context_mutates", "artifact_provides"))
        consumers = _entries_matching(all_entries, skip_key="", values=(value,), target_fields=("context_requires", "artifact_requires"))
        rows.append({"field": value, "producer_count": len(producers), "consumer_count": len(consumers), "producers": tuple(_entry_payload(item) for item in producers[:8]), "consumers": tuple(_entry_payload(item) for item in consumers[:8])})
    return tuple(rows)


def _relation_cards(neighbors: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[dict[str, Any], ...]:
    cards = []
    for name, label in _RELATION_LABELS.items():
        rows = tuple(neighbors.get(name, ()))
        cards.append({"group": name, "label": label, "count": len(rows), "items": rows[:8]})
    return tuple(cards)


def _entries_matching(entries: Sequence[CatalogEntry], *, skip_key: str, values: Sequence[str], target_fields: Sequence[str], artifact_only: bool = False) -> list[CatalogEntry]:
    wanted = {str(value).strip().lower() for value in values if str(value).strip()}
    if not wanted:
        return []
    out = []
    for candidate in entries:
        if skip_key and candidate.key == skip_key:
            continue
        candidate_fields = relation_fields(candidate)
        current: set[str] = set()
        for field_name in target_fields:
            for value in candidate_fields.get(field_name, ()):
                if artifact_only and not _looks_like_artifact(value):
                    continue
                current.add(str(value).strip().lower())
        if wanted.intersection(current):
            out.append(candidate)
    return sorted(out, key=lambda item: (item.kind, item.key))


def _role_companions(entry: CatalogEntry, *, all_entries: Sequence[CatalogEntry]) -> list[CatalogEntry]:
    required = {str(value).strip() for value in usage_profile(entry).get("required_roles", ()) if str(value).strip()}
    if not required:
        return []
    companions = [item for item in all_entries if item.key != entry.key and item.kind in required]
    return sorted(companions, key=lambda item: (item.kind, item.key))[:12]


def _linked_by(entry: CatalogEntry, *, all_entries: Sequence[CatalogEntry]) -> list[CatalogEntry]:
    out = []
    for candidate in all_entries:
        if candidate.key == entry.key:
            continue
        values = _values(candidate.metadata.get("companions") if isinstance(candidate.metadata, Mapping) else ())
        if entry.key in values:
            out.append(candidate)
    return sorted(out, key=lambda item: (item.kind, item.key))


def _entry_payload(entry: CatalogEntry) -> dict[str, Any]:
    return {"key": entry.key, "title": entry.title, "kind": entry.kind, "summary": entry.summary}


def _stage_for_kind(kind: str) -> str:
    raw = str(kind or "").strip()
    for stage in _FLOW_STAGES:
        if raw in stage["kinds"]:
            return str(stage["id"])
    return "trainer_lifecycle"


def _looks_like_artifact(value: str) -> bool:
    raw = str(value or "").strip().lower()
    return bool(raw and any(hint in raw for hint in _ARTIFACT_HINTS))


def _values(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else tuple()
    if isinstance(value, Mapping):
        return tuple(str(key).strip() for key in value.keys() if str(key).strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_values(item))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in out:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return tuple(deduped)
    text = str(value).strip()
    return (text,) if text else tuple()


def _unique(*groups: Any) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in _values(group):
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
    return tuple(out)


def _flatten_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else tuple()
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            out.append(str(key))
            out.extend(_flatten_strings(item))
        return tuple(out)
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return tuple(out)
    return (str(value),)


def _fmt_values(values: Any, *, limit: int = 6) -> str:
    normalized = _values(values)
    if not normalized:
        return "无显式字段"
    shown = tuple(normalized[:limit])
    suffix = "" if len(normalized) <= limit else f" 等 {len(normalized)} 项"
    return ", ".join(f"`{value}`" for value in shown) + suffix


def _merge_hint_dicts(*hints: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for hint in hints:
        for key, values in hint.items():
            buckets[str(key)].extend(_values(values))
    return {key: _unique(values) for key, values in buckets.items()}


def _entry_text(entry: CatalogEntry) -> str:
    return " ".join(_flatten_strings({"key": entry.key, "title": entry.title, "kind": entry.kind, "import_path": entry.import_path, "tags": tuple(entry.tags), "summary": entry.summary})).lower()


def _resolve_object(entry: CatalogEntry) -> Any | None:
    module_name, sep, attr_name = str(entry.import_path or "").partition(":")
    if not sep or not module_name or not attr_name:
        return None
    try:
        module = importlib.import_module(module_name)
        obj: Any = module
        for part in attr_name.split("."):
            obj = getattr(obj, part)
        return obj
    except Exception:
        return None


def _signature_config_keys(entry: CatalogEntry) -> tuple[str, ...]:
    obj = _resolve_object(entry)
    if obj is None:
        return tuple()
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        signature = inspect.signature(target)
    except Exception:
        return tuple()
    skip = {"self", "cls", "args", "kwargs", "context", "data", "state", "values", "inputs", "X", "y", "solver", "trainer"}
    keys: list[str] = []
    for name, param in signature.parameters.items():
        if name in skip or param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        keys.append(str(name))
    return tuple(keys)


def _public_operation_methods(entry: CatalogEntry) -> tuple[str, ...]:
    obj = _resolve_object(entry)
    if obj is None:
        return tuple()
    method_names = ("propose", "update", "init", "mutate", "repair", "encode", "decode", "parameter_layout", "init_values", "evaluate", "predict", "predict_proba", "predict_interval", "fit", "transform", "fit_transform", "compose", "build", "describe")
    return tuple(name for name in method_names if callable(getattr(obj, name, None)))


def _contract_summary_line(fields: Mapping[str, Sequence[str]]) -> str:
    pieces: list[str] = []
    if fields.get("context_requires"):
        pieces.append("读取 " + _fmt_values(fields.get("context_requires")))
    if fields.get("context_provides"):
        pieces.append("提供 " + _fmt_values(fields.get("context_provides")))
    if fields.get("context_mutates"):
        pieces.append("修改 " + _fmt_values(fields.get("context_mutates")))
    if fields.get("context_cache"):
        pieces.append("缓存 " + _fmt_values(fields.get("context_cache")))
    if fields.get("artifact_provides"):
        pieces.append("产出 artifact " + _fmt_values(fields.get("artifact_provides")))
    return "契约摘要：" + "；".join(pieces) + "。" if pieces else ""


__all__ = [
    "build_entry_relation_payload",
    "build_relation_payload_index",
    "flow_payload",
    "relation_fields",
    "relation_search_text",
    "usage_profile",
]
