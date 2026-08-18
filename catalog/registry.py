from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from blackbase.catalog import Catalog, CatalogEntry, load_catalog_paths

from blackbase.context import ContextContract
from blackbase.contracts import ComponentContract


_CATALOG_CACHE: Catalog | None = None


def get_catalog(*, refresh: bool = False) -> Catalog:
    global _CATALOG_CACHE
    if refresh or _CATALOG_CACHE is None:
        entries_by_key = {entry.key: entry for entry in _static_catalog_entries()}
        for entry in _backend_catalog_entries():
            entries_by_key[entry.key] = entry
        _CATALOG_CACHE = Catalog(_enrich_entries(entries_by_key.values()))
    return _CATALOG_CACHE


def _static_catalog_entries() -> tuple[CatalogEntry, ...]:
    builtin = Path(__file__).resolve().with_name("entries")
    entries_by_key = {entry.key: entry for entry in load_catalog_paths((builtin,))}
    raw_paths = os.environ.get("MLBLACK_CATALOG_PATH", "").strip()
    if raw_paths:
        paths = tuple(Path(item.strip().strip('"')) for item in raw_paths.split(os.pathsep) if item.strip())
        for entry in load_catalog_paths(paths):
            entries_by_key[entry.key] = entry
    return tuple(entries_by_key.values())


def enrich_catalog_entry(entry: CatalogEntry) -> CatalogEntry:
    """Attach a resolved context contract to a static catalog entry."""

    try:
        obj = _resolve_import_path(entry.import_path)
    except Exception as exc:
        return replace(
            entry,
            metadata={**dict(entry.metadata), "contract_error": repr(exc)},
        )
    payload = _contract_payload_for_object(obj)
    if not payload:
        return entry
    return replace(
        entry,
        contract={**dict(entry.contract), **payload},
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
    generated: list[CatalogEntry] = []
    module_name = str(getattr(module, "__name__", ""))
    for name, obj in inspect.getmembers(module):
        if not _is_catalog_component(obj, module_name=module_name):
            continue
        generated.append(_entry_from_object(obj, module_name=module_name, symbol_name=name))
    return tuple(generated)


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
