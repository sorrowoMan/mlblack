from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from mlblack.core.context_contracts import ContextContract
from mlblack.core.contracts import ComponentContract


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    title: str
    kind: str
    import_path: str
    tags: Sequence[str] = tuple()
    summary: str = ""
    contract: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "kind": self.kind,
            "import_path": self.import_path,
            "tags": list(self.tags),
            "summary": self.summary,
            "contract": dict(self.contract),
            "metadata": dict(self.metadata),
        }


class Catalog:
    def __init__(self, entries: Iterable[CatalogEntry]) -> None:
        self._entries = tuple(entries)
        self._by_key = {entry.key: entry for entry in self._entries}

    def list(self, *, kind: str | None = None, tag: str | None = None) -> tuple[CatalogEntry, ...]:
        entries = self._entries
        if kind is not None:
            entries = tuple(item for item in entries if item.kind == str(kind))
        if tag is not None:
            entries = tuple(item for item in entries if str(tag) in {str(x) for x in item.tags})
        return entries

    def search(self, query: str, *, kind: str | None = None, limit: int = 20) -> tuple[CatalogEntry, ...]:
        q = str(query).strip().lower()
        entries = self.list(kind=kind)
        if not q:
            return tuple(entries[: max(0, int(limit))])
        matched = [
            item for item in entries
            if q in item.key.lower()
            or q in item.title.lower()
            or q in item.summary.lower()
            or any(q in str(tag).lower() for tag in item.tags)
        ]
        return tuple(matched[: max(0, int(limit))])

    def get(self, key: str) -> CatalogEntry | None:
        return self._by_key.get(str(key))

    def show(self, key: str) -> CatalogEntry:
        item = self.get(str(key))
        if item is not None:
            return item
        raise KeyError(f"catalog entry not found: {key}")


_CATALOG_CACHE: Catalog | None = None


def get_catalog(*, refresh: bool = False) -> Catalog:
    global _CATALOG_CACHE
    if refresh or _CATALOG_CACHE is None:
        entries_by_key: dict[str, CatalogEntry] = {}
        for entry in _auto_discovered_entries():
            entries_by_key[entry.key] = entry
        for entry in _default_entries():
            entries_by_key[entry.key] = entry
        for entry in _backend_catalog_entries():
            entries_by_key[entry.key] = entry
        _CATALOG_CACHE = Catalog(_enrich_entries(entries_by_key.values()))
    return _CATALOG_CACHE


def enrich_catalog_entry(entry: CatalogEntry) -> CatalogEntry:
    """Attach a resolved context contract to a static catalog entry."""

    try:
        obj = _resolve_import_path(entry.import_path)
    except Exception as exc:
        return CatalogEntry(
            key=entry.key,
            title=entry.title,
            kind=entry.kind,
            import_path=entry.import_path,
            tags=tuple(entry.tags),
            summary=entry.summary,
            contract=dict(entry.contract),
            metadata={**dict(entry.metadata), "contract_error": repr(exc)},
        )
    payload = _contract_payload_for_object(obj)
    if not payload:
        return entry
    return CatalogEntry(
        key=entry.key,
        title=entry.title,
        kind=entry.kind,
        import_path=entry.import_path,
        tags=tuple(entry.tags),
        summary=entry.summary,
        contract={**dict(entry.contract), **payload},
        metadata=dict(entry.metadata),
    )


def _enrich_entries(entries: Iterable[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    return tuple(enrich_catalog_entry(entry) for entry in entries)


def _resolve_import_path(import_path: str) -> Any:
    module_name, sep, attr_name = str(import_path).partition(":")
    if not sep:
        return importlib.import_module(module_name)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def _contract_payload_for_object(obj: Any) -> dict[str, Any]:
    raw_contract = getattr(obj, "contract", None)
    has_context_attrs = any(
        hasattr(obj, attr)
        for attr in (
            "context_requires",
            "context_optional",
            "context_provides",
            "context_mutates",
            "context_cache",
            "requires_metrics",
            "metrics_fallback",
            "context_notes",
        )
    )
    if not has_context_attrs and not isinstance(raw_contract, (ComponentContract, ContextContract, Mapping)):
        return {}
    context_contract = ContextContract.from_component(obj, fallback_contract=raw_contract)
    if isinstance(raw_contract, ComponentContract):
        component_contract = ComponentContract.from_context_contract(
            context_contract,
            supports_gradient=raw_contract.supports_gradient,
            supports_batch=raw_contract.supports_batch,
            supports_resume=raw_contract.supports_resume,
            metadata=raw_contract.metadata,
        )
    else:
        component_contract = ComponentContract.from_context_contract(context_contract)
    payload = component_contract.describe()
    payload["unknown_context_keys"] = list(context_contract.unknown_keys())
    payload["unknown_metric_keys"] = list(context_contract.unknown_metric_keys())
    return payload


def _backend_catalog_entries() -> tuple[CatalogEntry, ...]:
    try:
        from mlblack.backends.catalog import list_backend_catalog_entries
    except Exception:
        return tuple()
    entries: list[CatalogEntry] = []
    for item in list_backend_catalog_entries():
        kind = str(item.get("kind", "backend"))
        name = str(item.get("name", "backend"))
        entries.append(
            CatalogEntry(
                key=f"{kind}.{name}",
                title=name,
                kind=kind,
                import_path="mlblack.backends.registry:get_backend",
                tags=("backend", str(item.get("backend", name)).split(".")[0]),
                summary=f"Backend catalog entry for {name}.",
                contract={
                    "provides": tuple(item.get("provides", ())),
                    "methods": dict(item.get("methods", {})),
                },
                metadata=dict(item.get("metadata", item)),
            )
        )
    return tuple(entries)


_DISCOVERY_PACKAGES: tuple[str, ...] = (
    "mlblack.adapters",
    "mlblack.assembly",
    "mlblack.bias",
    "mlblack.capabilities",
    "mlblack.core",
    "mlblack.integrations",
    "mlblack.models",
    "mlblack.pipeline",
    "mlblack.presets",
    "mlblack.problems",
    "mlblack.representations",
)

_KIND_SUFFIXES: tuple[str, ...] = (
    "Adapter",
    "Representation",
    "Problem",
    "Capability",
    "Bias",
    "Head",
    "Codec",
    "DataView",
    "Model",
    "Component",
    "Primitive",
    "Spec",
    "Config",
    "Contract",
)


def _auto_discovered_entries() -> tuple[CatalogEntry, ...]:
    flag = os.environ.get("MLBLACK_CATALOG_DISCOVERY", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return tuple()

    entries: dict[str, CatalogEntry] = {}
    for package_name in _DISCOVERY_PACKAGES:
        for module in _walk_modules(package_name):
            for entry in _catalog_entries_from_module(module):
                entries.setdefault(entry.key, entry)
    return tuple(entries.values())


def _walk_modules(package_name: str) -> Iterable[Any]:
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return
    yield package
    package_path = getattr(package, "__path__", None)
    if not package_path:
        return
    for modinfo in pkgutil.walk_packages(package_path, package.__name__ + "."):
        name = str(modinfo.name)
        if ".__pycache__" in name or name.endswith(".tests"):
            continue
        try:
            yield importlib.import_module(name)
        except Exception:
            continue


def _catalog_entries_from_module(module: Any) -> tuple[CatalogEntry, ...]:
    explicit = _explicit_catalog_entries(module)
    generated: list[CatalogEntry] = []
    module_name = str(getattr(module, "__name__", ""))
    for name, obj in inspect.getmembers(module):
        if not _is_catalog_component(obj, module_name=module_name):
            continue
        generated.append(_entry_from_object(obj, module_name=module_name, symbol_name=name))
    by_key = {entry.key: entry for entry in generated}
    for entry in explicit:
        by_key[entry.key] = entry
    return tuple(by_key.values())


def _explicit_catalog_entries(module: Any) -> tuple[CatalogEntry, ...]:
    raw = getattr(module, "CATALOG_ENTRIES", None)
    if not isinstance(raw, (list, tuple)):
        return tuple()
    entries: list[CatalogEntry] = []
    for item in raw:
        if isinstance(item, CatalogEntry):
            entries.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        try:
            entries.append(
                CatalogEntry(
                    key=str(item.get("key", "")).strip(),
                    title=str(item.get("title", "")).strip(),
                    kind=str(item.get("kind", "")).strip().lower(),
                    import_path=str(item.get("import_path", "")).strip(),
                    tags=_string_tuple(item.get("tags", ())),
                    summary=str(item.get("summary", "")).strip(),
                    contract=dict(item.get("contract", {}) or {}),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            )
        except Exception:
            continue
    return tuple(entry for entry in entries if entry.key and entry.kind and entry.import_path)


def _is_catalog_component(obj: Any, *, module_name: str) -> bool:
    if not inspect.isclass(obj):
        return False
    obj_module = str(getattr(obj, "__module__", ""))
    if obj_module != module_name:
        return False
    if str(getattr(obj, "__name__", "")).startswith("_"):
        return False
    if getattr(obj, "__module__", "").endswith(".contracts"):
        return bool(_class_has_component_contract(obj))
    return bool(_class_has_component_contract(obj) or _class_has_component_shape(obj))


def _class_has_component_contract(obj: Any) -> bool:
    if hasattr(obj, "contract"):
        return True
    return any(
        hasattr(obj, attr)
        for attr in (
            "context_requires",
            "context_optional",
            "context_provides",
            "context_mutates",
            "context_cache",
            "requires_metrics",
        )
    )


def _class_has_component_shape(obj: Any) -> bool:
    method_names = {
        "describe",
        "init",
        "mutate",
        "repair",
        "encode",
        "decode",
        "evaluate",
        "predict",
        "predict_proba",
        "fit",
        "transform",
        "apply",
    }
    return any(callable(getattr(obj, name, None)) for name in method_names)


def _entry_from_object(obj: Any, *, module_name: str, symbol_name: str) -> CatalogEntry:
    kind = _kind_for_module(module_name, symbol_name=symbol_name)
    key = f"{kind}.{_key_stem(symbol_name, kind=kind)}"
    tags = _tags_for_module(module_name, kind=kind)
    summary = _summary_for_object(obj, key=key)
    metadata = {
        "catalog_source": "auto_discovery",
        "architecture_path": _architecture_path(module_name, kind=kind),
        "module": module_name,
        "symbol": symbol_name,
    }
    return CatalogEntry(
        key=key,
        title=symbol_name,
        kind=kind,
        import_path=f"{module_name}:{symbol_name}",
        tags=tags,
        summary=summary,
        metadata=metadata,
    )


def _kind_for_module(module_name: str, *, symbol_name: str) -> str:
    if ".adapters" in module_name:
        return "adapter"
    if ".representations.heads" in module_name:
        return "head"
    if ".representations.codecs" in module_name:
        return "codec"
    if ".representations" in module_name:
        return "representation"
    if ".problems" in module_name:
        return "problem_bridge" if "proxy" in module_name else "problem"
    if ".pipeline.data_views" in module_name:
        return "data_view"
    if ".pipeline.numericizer" in module_name:
        return "numericizer"
    if ".pipeline.conditional" in module_name:
        return "conditional"
    if ".pipeline.symbolic" in module_name:
        return "symbolic_pipeline"
    if ".pipeline" in module_name:
        return "pipeline"
    if ".bias" in module_name:
        return "bias"
    if ".capabilities" in module_name:
        return "capability"
    if ".models" in module_name:
        if symbol_name.endswith("Provider"):
            return "provider"
        return "model"
    if ".assembly" in module_name:
        return "assembly"
    if ".integrations" in module_name:
        if "nsgablack_symbolic" in module_name:
            return "nsgablack_symbolic"
        if "nsgablack_neural" in module_name:
            return "nsgablack_neural"
        return "integration"
    if ".presets" in module_name:
        return "preset"
    if ".core" in module_name and symbol_name.endswith("Trainer"):
        return "trainer"
    if ".core" in module_name:
        return "core"
    return "component"


def _key_stem(symbol_name: str, *, kind: str) -> str:
    stem = str(symbol_name)
    if kind == "provider" and stem.endswith("Provider") and len(stem) > len("Provider"):
        stem = stem[: -len("Provider")]
        return _camel_to_snake(stem)
    for suffix in _KIND_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            stem = stem[: -len(suffix)]
            break
    if kind == "problem_bridge" and stem.endswith("Proxy"):
        stem = stem[:-5]
    return _camel_to_snake(stem)


def _camel_to_snake(value: str) -> str:
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(value))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return text.strip("_").lower() or "component"


def _tags_for_module(module_name: str, *, kind: str) -> tuple[str, ...]:
    parts = tuple(part for part in module_name.split(".")[1:] if part)
    tags = ["auto", kind, *parts]
    return tuple(dict.fromkeys(tags))


def _architecture_path(module_name: str, *, kind: str) -> str:
    namespace = module_name.removeprefix("mlblack.")
    if kind in {"head", "codec", "representation"}:
        return f"representation/{namespace}"
    if kind == "data_view":
        tail = namespace.removeprefix("pipeline.data_views").strip(".")
        return "pipeline/data_views" if not tail else f"pipeline/data_views/{tail.replace('.', '/')}"
    if kind in {"numericizer", "conditional", "symbolic_pipeline", "pipeline"}:
        return f"pipeline/{namespace}"
    if kind.startswith("nsgablack_"):
        return f"integration/{namespace}"
    return namespace


def _summary_for_object(obj: Any, *, key: str) -> str:
    doc = inspect.getdoc(obj) or ""
    if doc:
        return doc.splitlines()[0].strip()
    return f"Auto-discovered mlblack component `{key}`."


def _string_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else tuple()
    if isinstance(values, Mapping):
        return tuple(str(key).strip() for key in values.keys() if str(key).strip())
    if isinstance(values, (list, tuple, set, frozenset)):
        out: list[str] = []
        for value in values:
            out.extend(_string_tuple(value))
        return tuple(out)
    text = str(values).strip()
    return (text,) if text else tuple()


def _default_entries() -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            key="example.etf_temporal_forecast",
            title="ETF Temporal Forecast",
            kind="example",
            import_path="mlblack.integrations.etf_temporal_forecast:run_etf_temporal_forecast_multi_seed",
            tags=("example", "etf", "forecast", "walk-forward", "rank-ic", "portfolio", "scaffold"),
            summary="ETF walk-forward temporal forecast case with ridge/boosting baselines and rank-IC, hit-rate, Sharpe, drawdown and turnover metrics.",
            metadata={
                "example_entry": "examples/cases/etf_temporal_forecast/START_HERE.md",
                "dataset": "runs/etf_temporal_forecast/cache/multi_etf_returns_momodel_kaggle.parquet",
            },
        ),
        CatalogEntry(
            key="example.traffic_arimax_attribution",
            title="ARIMAX Factor Attribution on Traffic CI",
            kind="example",
            import_path="examples.cases.traffic_congestion.arimax_factor_attribution:run_trainer",
            tags=("example", "traffic", "arimax", "attribution", "time-series"),
            summary="ARIMAX(2,0,1) factor group attribution decomposing CI variation into Weather/AQI/Holiday/CI_Lags/Time_Cyclic groups.",
            metadata={
                "example_entry": "examples/cases/traffic_congestion/arimax_factor_attribution/START_HERE.md",
            },
        ),
        CatalogEntry(
            key="example.traffic_gam_linearity",
            title="GAM Nonlinearity Check on Traffic CI",
            kind="example",
            import_path="examples.cases.traffic_congestion.gam_linearity_check:run_trainer",
            tags=("example", "traffic", "gam", "nonlinear", "diagnostic"),
            summary="B-spline GAM vs linear regression comparison diagnosing whether linear assumptions hold for traffic CI prediction.",
            metadata={
                "example_entry": "examples/cases/traffic_congestion/gam_linearity_check/START_HERE.md",
            },
        ),
        CatalogEntry(
            key="example.traffic_shap_consistency",
            title="SHAP Contribution Consistency on Traffic CI",
            kind="example",
            import_path="examples.cases.traffic_congestion.shap_contribution_check:run_trainer",
            tags=("example", "traffic", "shap", "xgboost", "attribution", "diagnostic"),
            summary="Cross-paradigm feature importance comparison: Linear coefficients vs XGBoost gain vs SHAP vs Permutation on traffic CI.",
            metadata={
                "example_entry": "examples/cases/traffic_congestion/shap_contribution_check/START_HERE.md",
            },
        ),
        CatalogEntry(
            key="example.traffic_granger_causality",
            title="Granger Causality Check on Traffic CI",
            kind="example",
            import_path="examples.cases.traffic_congestion.granger_causality_check:run_trainer",
            tags=("example", "traffic", "granger", "causal", "diagnostic"),
            summary="Pairwise Granger causality tests between CI and external factors (weather, AQI, holidays) on real traffic data.",
            metadata={
                "example_entry": "examples/cases/traffic_congestion/granger_causality_check/START_HERE.md",
            },
        ),
        CatalogEntry(
            key="example.traffic_xgboost_baseline",
            title="XGBoost Baseline for Traffic CI Prediction",
            kind="example",
            import_path="examples.cases.traffic_congestion.xgboost_baseline:run_trainer",
            tags=("example", "traffic", "xgboost", "tree", "prediction"),
            summary="XGBoost baseline on traffic CI forward prediction (lag-only features), using mlblack tree_boosting preset.",
            metadata={
                "example_entry": "examples/cases/traffic_congestion/xgboost_baseline/START_HERE.md",
            },
        ),
        CatalogEntry(
            key="example.traffic_symbolic_regression",
            title="Symbolic Regression on Traffic CI",
            kind="example",
            import_path="examples.cases.traffic_congestion.symbolic_regression:run_trainer",
            tags=("example", "traffic", "symbolic", "regression", "linear", "gradient_descent"),
            summary="Gradient-descent linear regression on traffic CI mechanism reconstruction (same-day features), baseline for symbolic approach.",
            metadata={
                "example_entry": "examples/cases/traffic_congestion/symbolic_regression/START_HERE.md",
            },
        ),
        CatalogEntry(
            key="trainer.composable",
            title="ComposableTrainer",
            kind="trainer",
            import_path="mlblack.core.trainer:ComposableTrainer",
            tags=("control-plane", "solver-like"),
            summary="Trainer control plane with mounted OptimizerAdapter.",
        ),
        CatalogEntry(
            key="adapter.gradient_descent",
            title="GradientDescentAdapter",
            kind="adapter",
            import_path="mlblack.adapters.gradient_descent:GradientDescentAdapter",
            tags=("gradient", "linear", "resume"),
            summary="Consumes feedback.gradients and updates UnknownState.",
        ),
        CatalogEntry(
            key="adapter.functional_backprop",
            title="FunctionalBackpropAdapter",
            kind="adapter",
            import_path="mlblack.adapters.functional_backprop:FunctionalBackpropAdapter",
            tags=("gradient", "neural", "functional-backend", "resume"),
            summary="Uses problem-owned functional gradients plus backend optimizer.sgd_step.",
        ),
        CatalogEntry(
            key="adapter.random_search",
            title="RandomSearchAdapter",
            kind="adapter",
            import_path="mlblack.adapters.random_search:RandomSearchAdapter",
            tags=("black-box", "interval", "resume"),
            summary="Black-box candidate search for non-gradient heads.",
        ),
        CatalogEntry(
            key="adapter.estimator_spec_search",
            title="EstimatorSpecSearchAdapter",
            kind="adapter",
            import_path="mlblack.adapters.estimator_search:EstimatorSpecSearchAdapter",
            tags=("tree", "xgboost", "sklearn", "resume"),
            summary="Searches decoded external estimator specs.",
        ),
        CatalogEntry(
            key="adapter.torch_backprop",
            title="TorchBackpropAdapter",
            kind="adapter",
            import_path="mlblack.adapters.torch_backprop:TorchBackpropAdapter",
            tags=("neural", "torch", "resource-context", "resume"),
            summary="Torch gradient engine for parameter-vector MLP representations.",
        ),
        CatalogEntry(
            key="representation.orthogonal_linear",
            title="OrthogonalPointLinearRepresentation",
            kind="representation",
            import_path="mlblack.representations.orthogonal_point:OrthogonalPointLinearRepresentation",
            tags=("linear", "orthogonal", "head"),
            summary="Unknown vector decoded through orthogonal feature map and output head.",
        ),
        CatalogEntry(
            key="representation.estimator_spec",
            title="EstimatorSpecRepresentation",
            kind="representation",
            import_path="mlblack.representations.estimator_specs:EstimatorSpecRepresentation",
            tags=("tree", "xgboost", "sklearn", "codec"),
            summary="Unknown vector decoded into a typed external estimator spec.",
        ),
        CatalogEntry(
            key="representation.numpy_mlp",
            title="NumpyMLPPointRepresentation",
            kind="representation",
            import_path="mlblack.representations.neural_mlp:NumpyMLPPointRepresentation",
            tags=("neural", "torch", "head", "codec"),
            summary="Flat parameter vector decoded into a numpy MLP model.",
        ),
        CatalogEntry(
            key="codec.neural_temporal_lstm_spec",
            title="NeuralGraphSpec.temporal_lstm",
            kind="codec",
            import_path="mlblack.representations.codecs.neural.specs:NeuralGraphSpec.temporal_lstm",
            tags=("neural", "time-series", "temporal", "lstm", "backend-route"),
            summary="构建 LSTM 风格序列预测的 NeuralGraphSpec 路由，通过后端神经网络降层执行。",
        ),
        CatalogEntry(
            key="codec.neural_temporal_tcn_spec",
            title="NeuralGraphSpec.temporal_tcn",
            kind="codec",
            import_path="mlblack.representations.codecs.neural.specs:NeuralGraphSpec.temporal_tcn",
            tags=("neural", "time-series", "temporal", "tcn", "backend-route"),
            summary="构建 TCN 风格序列预测的 NeuralGraphSpec 路由，通过后端神经网络降层执行。",
        ),
        CatalogEntry(
            key="codec.neural_temporal_transformer_spec",
            title="NeuralGraphSpec.temporal_transformer",
            kind="codec",
            import_path="mlblack.representations.codecs.neural.specs:NeuralGraphSpec.temporal_transformer",
            tags=("neural", "time-series", "temporal", "transformer", "backend-route"),
            summary="构建时序 Transformer 预测的 NeuralGraphSpec 路由，通过后端神经网络降层执行。",
        ),
        CatalogEntry(
            key="codec.neural_temporal_nbeats_spec",
            title="NeuralGraphSpec.temporal_nbeats",
            kind="codec",
            import_path="mlblack.representations.codecs.neural.specs:NeuralGraphSpec.temporal_nbeats",
            tags=("neural", "time-series", "temporal", "nbeats", "backend-route"),
            summary="构建 N-BEATS 风格堆叠残差块预测的 NeuralGraphSpec 路由，通过后端神经网络降层执行。",
        ),
        CatalogEntry(
            key="codec.neural_temporal_deepar_spec",
            title="NeuralGraphSpec.temporal_deepar",
            kind="codec",
            import_path="mlblack.representations.codecs.neural.specs:NeuralGraphSpec.temporal_deepar",
            tags=("neural", "time-series", "temporal", "deepar", "probabilistic", "backend-route"),
            summary="构建 DeepAR 风格概率预测的 NeuralGraphSpec 路由，LSTM 主干输出 mu 和 log_sigma 分布参数。",
        ),
        CatalogEntry(
            key="codec.neural_temporal_patchtst_spec",
            title="NeuralGraphSpec.temporal_patchtst",
            kind="codec",
            import_path="mlblack.representations.codecs.neural.specs:NeuralGraphSpec.temporal_patchtst",
            tags=("neural", "time-series", "temporal", "patchtst", "transformer", "backend-route"),
            summary="构建 PatchTST 风格补丁嵌入 + Transformer 编码器预测的 NeuralGraphSpec 路由。",
        ),
        CatalogEntry(
            key="codec.neural_temporal_tft_spec",
            title="NeuralGraphSpec.temporal_tft",
            kind="codec",
            import_path="mlblack.representations.codecs.neural.specs:NeuralGraphSpec.temporal_tft",
            tags=("neural", "time-series", "temporal", "tft", "attention", "backend-route"),
            summary="构建 Temporal Fusion Transformer (TFT) 风格的 NeuralGraphSpec 路由，含门控残差网络和可解释多头自注意力。",
        ),
        CatalogEntry(
            key="problem.supervised_regression",
            title="SupervisedRegressionProblem",
            kind="problem",
            import_path="mlblack.problems.supervised:SupervisedRegressionProblem",
            tags=("regression", "gradient", "residuals"),
            summary="Data-dependent evaluator for point regression.",
        ),
        CatalogEntry(
            key="model.integrated_prediction",
            title="IntegratedPredictionModel",
            kind="model",
            import_path="mlblack.models.composition:IntegratedPredictionModel",
            tags=("composition", "integration", "residual", "stacking"),
            summary="Combines named fitted model predictions without owning training orchestration.",
        ),
        CatalogEntry(
            key="model.prediction_integration_component",
            title="PredictionIntegrationComponent",
            kind="model",
            import_path="mlblack.models.composition:PredictionIntegrationComponent",
            tags=("composition", "integration", "residual", "stacking", "component"),
            summary="Builds IntegratedPredictionModel from named fitted component models.",
        ),
        CatalogEntry(
            key="pipeline.data_pipeline_chain",
            title="DataPipeline",
            kind="pipeline",
            import_path="mlblack.pipeline.base:DataPipeline",
            tags=("pipeline", "data", "chain", "fit-transform"),
            summary="Ordered data preparation chain for a trainer assembly.",
        ),
        CatalogEntry(
            key="pipeline.model_conditioned_target",
            title="ModelConditionedTargetComponent",
            kind="pipeline",
            import_path="mlblack.pipeline.model_conditioning:ModelConditionedTargetComponent",
            tags=("pipeline", "residual", "model-conditioned", "stage-surface"),
            summary="Builds next-stage targets by calling an existing model, e.g. y - model.predict(X).",
        ),
        CatalogEntry(
            key="pipeline.model_conditioned_target_config",
            title="ModelConditionedTargetConfig",
            kind="pipeline",
            import_path="mlblack.pipeline.model_conditioning:ModelConditionedTargetConfig",
            tags=("pipeline", "residual", "model-conditioned", "config"),
            summary="Configuration for building a next-stage target from a reference model.",
        ),
        CatalogEntry(
            key="problem.supervised_estimator_fit",
            title="SupervisedEstimatorFitRegressionProblem",
            kind="problem",
            import_path="mlblack.problems.supervised:SupervisedEstimatorFitRegressionProblem",
            tags=("tree", "xgboost", "artifact"),
            summary="Fits decoded estimator specs and scores the fitted estimator.",
        ),
        CatalogEntry(
            key="capability.checkpoint",
            title="CheckpointCapability",
            kind="capability",
            import_path="mlblack.capabilities.checkpoint:CheckpointCapability",
            tags=("state", "resume", "snapshot"),
            summary="Writes trainer state snapshots during fit.",
        ),
        CatalogEntry(
            key="capability.experiment_tracker",
            title="ExperimentTrackerCapability",
            kind="capability",
            import_path="mlblack.capabilities.tracking:ExperimentTrackerCapability",
            tags=("experiment", "sqlite", "run-record"),
            summary="Records fit/step/evaluation events to an experiment store.",
        ),
        CatalogEntry(
            key="capability.resource_audit",
            title="ResourceAuditCapability",
            kind="capability",
            import_path="mlblack.capabilities.resource_audit:ResourceAuditCapability",
            tags=("l0", "resource", "audit"),
            summary="Audits effective ResourceContext during fit.",
        ),
        CatalogEntry(
            key="bias.state_l2",
            title="StateL2Bias",
            kind="bias",
            import_path="mlblack.bias.policies:StateL2Bias",
            tags=("soft-preference", "regularization"),
            summary="Adds a soft L2 penalty to feedback objectives.",
        ),
        CatalogEntry(
            key="bias.objective_weight",
            title="ObjectiveWeightBias",
            kind="bias",
            import_path="mlblack.bias.policies:ObjectiveWeightBias",
            tags=("soft-preference", "multi-objective"),
            summary="Reweights objective dimensions before adapter update.",
        ),
        CatalogEntry(
            key="representation.piecewise",
            title="PiecewiseRepresentation",
            kind="representation",
            import_path="mlblack.representations.conditional:PiecewiseRepresentation",
            tags=("conditional", "piecewise", "router"),
            summary="Concatenates branch states and decodes a routed piecewise model.",
        ),
        CatalogEntry(
            key="problem.supervised_classification",
            title="SupervisedClassificationProblem",
            kind="problem",
            import_path="mlblack.problems.classification:SupervisedClassificationProblem",
            tags=("classification", "probability", "log-loss"),
            summary="Evaluator for classification accuracy and log-loss objectives.",
        ),
        CatalogEntry(
            key="assembly.build_trainer",
            title="build_trainer",
            kind="assembly",
            import_path="mlblack.assembly.builders:build_trainer",
            tags=("scaffold", "trainer", "inner-training"),
            summary="Builds one inner ML trainer; orchestration is delegated to nsgablack.",
        ),
        CatalogEntry(
            key="schema.scaffold_config",
            title="ScaffoldConfig",
            kind="schema",
            import_path="mlblack.assembly.schema.spec:ScaffoldConfig",
            tags=("schema", "config", "scaffold"),
            summary="Top-level JSON-compatible scaffold contract.",
        ),
        CatalogEntry(
            key="problem.training_proxy",
            title="MLBlackTrainingProxy",
            kind="problem_bridge",
            import_path="mlblack.problems.proxy:MLBlackTrainingProxy",
            tags=("cross-framework", "proxy", "training-contract"),
            summary="Framework-neutral proxy for outer optimizers invoking mlblack inner training.",
        ),
        CatalogEntry(
            key="numericizer.default",
            title="DefaultNumericizer",
            kind="numericizer",
            import_path="mlblack.pipeline.numericizer.default:DefaultNumericizer",
            tags=("data", "schema", "feature-space"),
            summary="Converts schema-backed raw rows into NumericDataView.",
        ),
        CatalogEntry(
            key="pipeline.feature_space",
            title="FeatureSpaceComponent",
            kind="pipeline",
            import_path="mlblack.pipeline.feature_space:FeatureSpaceComponent",
            tags=("pipeline", "feature-space", "metadata"),
            summary="Records feature-space metadata in the data pipeline.",
        ),
        CatalogEntry(
            key="conditional.primitives",
            title="Conditional Primitives",
            kind="conditional",
            import_path="mlblack.pipeline.conditional.primitives:ConditionalPrimitive",
            tags=("binary-gate", "soft-gate", "hinge", "onehot"),
            summary="Reusable conditional feature and routing primitives.",
        ),
        CatalogEntry(
            key="conditional.composer",
            title="PrimitiveFeatureComposer",
            kind="conditional",
            import_path="mlblack.pipeline.conditional.composer:PrimitiveFeatureComposer",
            tags=("composer", "feature-engineering", "piecewise"),
            summary="Composes conditional primitives into deterministic feature transforms.",
        ),
        CatalogEntry(
            key="bias.dynamic_pool",
            title="DynamicPoolBias",
            kind="bias",
            import_path="mlblack.bias.policies:DynamicPoolBias",
            tags=("soft-preference", "pool", "event"),
            summary="Projects a context-dependent candidate/model pool hint.",
        ),
        CatalogEntry(
            key="bias.branch_policy",
            title="BranchPolicyBias",
            kind="bias",
            import_path="mlblack.bias.policies:BranchPolicyBias",
            tags=("soft-preference", "conditional", "branch"),
            summary="Exposes branch preferences for conditional/piecewise representations.",
        ),
        CatalogEntry(
            key="bias.objective_policy",
            title="ObjectivePolicyBias",
            kind="bias",
            import_path="mlblack.bias.policies:ObjectivePolicyBias",
            tags=("soft-preference", "multi-objective", "policy"),
            summary="Context-aware objective reweighting policy.",
        ),
        CatalogEntry(
            key="dashboard.catalog_html",
            title="catalog dashboard export",
            kind="dashboard",
            import_path="mlblack.catalog.dashboard:export_catalog_html",
            tags=("catalog", "html", "report"),
            summary="Exports a lightweight HTML catalog report.",
        ),
        CatalogEntry(
            key="dashboard.experiment_html",
            title="experiment dashboard export",
            kind="dashboard",
            import_path="mlblack.catalog.experiment.dashboard:export_experiment_html",
            tags=("experiment", "html", "report"),
            summary="Exports a lightweight HTML experiment record report.",
        ),
        CatalogEntry(
            key="dashboard.artifact_html",
            title="artifact dashboard export",
            kind="dashboard",
            import_path="mlblack.catalog.artifacts:export_artifact_html",
            tags=("artifact", "symbolic", "html", "report"),
            summary="Exports a static HTML viewer for typed mlblack artifacts.",
        ),
        CatalogEntry(
            key="dashboard.backend_matrix_html",
            title="backend capability matrix export",
            kind="dashboard",
            import_path="mlblack.catalog.backend_dashboard:export_backend_matrix_html",
            tags=("backend", "capability", "matrix", "html", "report"),
            summary="Exports a static HTML matrix of backend capability support.",
        ),
        CatalogEntry(
            key="dashboard.catalog_web",
            title="catalog query web app",
            kind="dashboard",
            import_path="mlblack.catalog.web_app:serve_catalog_web",
            tags=("catalog", "web", "query", "db-only"),
            summary="Serves a lightweight catalog query UI and JSON API without external web dependencies.",
        ),
        CatalogEntry(
            key="dashboard.catalog_streamlit",
            title="catalog Streamlit dashboard",
            kind="dashboard",
            import_path="mlblack.catalog.dashboard:launch_catalog_dashboard",
            tags=("catalog", "streamlit", "web", "dashboard", "db-only"),
            summary="Launches an interactive Streamlit catalog dashboard similar to the nsgablack catalog UI.",
        ),
        CatalogEntry(
            key="head.binary_logistic",
            title="BinaryLogisticHead",
            kind="head",
            import_path="mlblack.representations.heads.probability:BinaryLogisticHead",
            tags=("classification", "probability", "logistic"),
            summary="Wraps a scalar logit decoder as binary predict_proba output.",
        ),
        CatalogEntry(
            key="head.softmax",
            title="SoftmaxHead",
            kind="head",
            import_path="mlblack.representations.heads.probability:SoftmaxHead",
            tags=("classification", "probability", "multiclass"),
            summary="Allocates one base decoder block per class and returns softmax probabilities.",
        ),
        CatalogEntry(
            key="head.piecewise",
            title="PiecewiseHead",
            kind="head",
            import_path="mlblack.representations.heads.conditional:PiecewiseHead",
            tags=("conditional", "piecewise", "branch"),
            summary="Allocates one base decoder block per branch and returns a PiecewiseModel.",
        ),
        CatalogEntry(
            key="preset.orthogonal_logistic_classification",
            title="orthogonal logistic classification preset",
            kind="preset",
            import_path="mlblack.presets.classification:build_orthogonal_logistic_classification_trainer",
            tags=("classification", "logistic", "orthogonal"),
            summary="Orthogonal linear logits with binary probability head and classification metrics.",
        ),
        # ── time series presets ──
        CatalogEntry(
            key="preset.temporal_lstm_forecast",
            title="temporal LSTM forecast preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_temporal_lstm_forecast_trainer",
            tags=("neural", "time-series", "temporal", "lstm", "preset"),
            summary="LSTM 神经序列预测 Trainer 预设，搭配 NeuralGraphBackpropAdapter。",
        ),
        CatalogEntry(
            key="preset.temporal_tcn_forecast",
            title="temporal TCN forecast preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_temporal_tcn_forecast_trainer",
            tags=("neural", "time-series", "temporal", "tcn", "preset"),
            summary="TCN 神经序列预测 Trainer 预设，搭配 NeuralGraphBackpropAdapter。",
        ),
        CatalogEntry(
            key="preset.temporal_transformer_forecast",
            title="temporal Transformer forecast preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_temporal_transformer_forecast_trainer",
            tags=("neural", "time-series", "temporal", "transformer", "preset"),
            summary="Transformer 神经序列预测 Trainer 预设，搭配 NeuralGraphBackpropAdapter。",
        ),
        CatalogEntry(
            key="preset.temporal_nbeats_forecast",
            title="temporal N-BEATS forecast preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_temporal_nbeats_forecast_trainer",
            tags=("neural", "time-series", "temporal", "nbeats", "preset"),
            summary="N-BEATS 风格堆叠残差块预测 Trainer 预设，搭配 NeuralGraphBackpropAdapter。",
        ),
        CatalogEntry(
            key="preset.temporal_deepar_forecast",
            title="temporal DeepAR forecast preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_temporal_deepar_forecast_trainer",
            tags=("neural", "time-series", "temporal", "deepar", "probabilistic", "preset"),
            summary="DeepAR 风格概率预测 Trainer 预设，LSTM 主干输出高斯分布参数，搭配 NeuralGraphBackpropAdapter。",
        ),
        CatalogEntry(
            key="preset.temporal_patchtst_forecast",
            title="temporal PatchTST forecast preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_temporal_patchtst_forecast_trainer",
            tags=("neural", "time-series", "temporal", "patchtst", "transformer", "preset"),
            summary="PatchTST 补丁嵌入 + Transformer 编码器预测 Trainer 预设，搭配 NeuralGraphBackpropAdapter。",
        ),
        CatalogEntry(
            key="preset.temporal_tft_forecast",
            title="temporal TFT forecast preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_temporal_tft_forecast_trainer",
            tags=("neural", "time-series", "temporal", "tft", "attention", "preset"),
            summary="Temporal Fusion Transformer (TFT) 预测 Trainer 预设，含门控残差网络和可解释多头自注意力。",
        ),
        CatalogEntry(
            key="codec.neural_tabular_tabnet_spec",
            title="NeuralGraphSpec.tabular_tabnet",
            kind="codec",
            import_path="mlblack.representations.codecs.neural.specs:NeuralGraphSpec.tabular_tabnet",
            tags=("neural", "tabular", "tabnet", "attention", "backend-route"),
            summary="构建 TabNet 风格表格深度学习的 NeuralGraphSpec 路由，含特征 Transformer、注意力掩码和 GLU 激活。",
        ),
        CatalogEntry(
            key="preset.tabular_tabnet_classification",
            title="tabular TabNet classification preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_tabular_tabnet_classification_trainer",
            tags=("neural", "tabular", "tabnet", "classification", "preset"),
            summary="TabNet 表格分类 Trainer 预设，搭配 NeuralGraphBackpropAdapter 和 SupervisedClassificationProblem。",
        ),
        CatalogEntry(
            key="preset.tabular_tabnet_regression",
            title="tabular TabNet regression preset",
            kind="preset",
            import_path="mlblack.presets.neural:build_tabular_tabnet_regression_trainer",
            tags=("neural", "tabular", "tabnet", "regression", "preset"),
            summary="TabNet 表格回归 Trainer 预设，搭配 NeuralGraphBackpropAdapter 和 SupervisedRegressionProblem。",
        ),
        CatalogEntry(
            key="preset.baseline_forecast_search",
            title="baseline forecast search preset",
            kind="preset",
            import_path="mlblack.presets.time_series:build_baseline_forecast_search_trainer",
            tags=("time-series", "baseline", "naive", "random-search", "preset"),
            summary="对 naive/季节性-naive/移动平均预测策略进行随机搜索。",
        ),
        CatalogEntry(
            key="preset.linear_autoregressive_forecast",
            title="linear autoregressive forecast preset",
            kind="preset",
            import_path="mlblack.presets.time_series:build_linear_autoregressive_forecast_trainer",
            tags=("time-series", "linear", "autoregressive", "preset"),
            summary="预拟合线性自回归预测模型，搭配 FixedForecastModelRepresentation。",
        ),
        CatalogEntry(
            key="preset.arima_sarimax_forecast",
            title="ARIMA/SARIMAX forecast preset",
            kind="preset",
            import_path="mlblack.presets.time_series:build_arima_sarimax_forecast_trainer",
            tags=("time-series", "arima", "sarimax", "preset"),
            summary="ARIMA/SARIMAX 预测模型，通过 ARIMASARIMAXProvider 搭配 FixedForecastModelRepresentation。",
        ),
        CatalogEntry(
            key="catalog.query",
            title="catalog query",
            kind="catalog",
            import_path="mlblack.catalog.query:query_catalog",
            tags=("catalog", "facet", "deep-link"),
            summary="Search catalog entries with facets and deep-link payload.",
        ),
        CatalogEntry(
            key="catalog.query_db",
            title="DB-only catalog query",
            kind="catalog",
            import_path="mlblack.catalog.query:query_catalog_db",
            tags=("catalog", "sqlite", "db-only", "facet"),
            summary="Query a materialized catalog DB without registry fallback.",
        ),
        CatalogEntry(
            key="catalog.materialize_db",
            title="materialize catalog DB",
            kind="catalog",
            import_path="mlblack.catalog.store:materialize_catalog_db",
            tags=("catalog", "sqlite", "materialize"),
            summary="Materializes the current registry/discovery catalog into a SQLite DB snapshot.",
        ),
        CatalogEntry(
            key="catalog.store_sqlite",
            title="SQLiteCatalogStore",
            kind="catalog",
            import_path="mlblack.catalog.store:SQLiteCatalogStore",
            tags=("catalog", "sqlite", "db-only"),
            summary="SQLite implementation of the DB-only catalog store surface.",
        ),
        CatalogEntry(
            key="catalog.store_postgresql",
            title="PostgresCatalogStore",
            kind="catalog",
            import_path="mlblack.catalog.store:PostgresCatalogStore",
            tags=("catalog", "postgresql", "db-only"),
            summary="PostgreSQL implementation of the DB-only catalog store surface.",
        ),
        CatalogEntry(
            key="catalog.resolve_store",
            title="resolve_catalog_store",
            kind="catalog",
            import_path="mlblack.catalog.store:resolve_catalog_store",
            tags=("catalog", "sqlite", "postgresql", "db-only"),
            summary="Resolves SQLite or PostgreSQL catalog stores from URL/path/env configuration.",
        ),
        CatalogEntry(
            key="experiment.query",
            title="experiment query",
            kind="experiment",
            import_path="mlblack.catalog.experiment.query:query_experiments",
            tags=("experiment", "sqlite", "facet"),
            summary="Query SQLite experiment records with simple filters and facets.",
        ),
        CatalogEntry(
            key="pipeline.source.s3_parquet",
            title="S3ParquetSource",
            kind="pipeline",
            import_path="mlblack.pipeline.sources:S3ParquetSource",
            tags=("pipeline", "source", "s3", "parquet", "cloud"),
            summary="Read Parquet datasets from S3 or S3-compatible stores (MinIO) into a NumericDataView.",
        ),
        CatalogEntry(
            key="pipeline.source.hive_query",
            title="HiveQuerySource",
            kind="pipeline",
            import_path="mlblack.pipeline.sources:HiveQuerySource",
            tags=("pipeline", "source", "hive", "sql", "cloud"),
            summary="Execute SQL against HiveServer2 and materialise results into a NumericDataView.",
        ),
        CatalogEntry(
            key="pipeline.source.jdbc_query",
            title="JDBCQuerySource",
            kind="pipeline",
            import_path="mlblack.pipeline.sources:JDBCQuerySource",
            tags=("pipeline", "source", "jdbc", "sql", "relational"),
            summary="Generic JDBC query source — Postgres, MySQL, Oracle, etc. — into NumericDataView.",
        ),
    )


