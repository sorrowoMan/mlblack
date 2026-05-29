from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mlblack.core.context_contracts import ContextContract
from mlblack.core.context_keys import METRIC_FALLBACKS, METRIC_KEYS
from mlblack.core.contracts import ComponentContract


@dataclass(frozen=True)
class DoctorDiagnostic:
    level: str
    code: str
    message: str
    path: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class DoctorReport:
    project_root: Path
    diagnostics: tuple[DoctorDiagnostic, ...]

    @property
    def ok(self) -> bool:
        return not any(item.level == "error" for item in self.diagnostics)


def run_project_doctor(path: str | Path | None = None, *, strict: bool = False) -> DoctorReport:
    root = Path(path or Path.cwd()).resolve()
    package_root = root / "mlblack" if (root / "mlblack").is_dir() else root
    diags: list[DoctorDiagnostic] = []

    _require_files(
        package_root,
        diags,
        (
            "core/trainer.py",
            "core/adapter.py",
            "core/representation.py",
            "core/problem.py",
            "core/resources.py",
            "core/state.py",
            "core/artifacts.py",
            "core/context_keys.py",
            "core/context_contracts.py",
            "core/contracts.py",
            "assembly/spec.py",
            "assembly/builders.py",
            "problems/training/task.py",
            "problems/proxy.py",
            "pipeline/numericizer/plan.py",
            "pipeline/feature_space.py",
            "pipeline/conditional/primitives.py",
            "representations/heads/probability.py",
            "representations/heads/conditional.py",
            "problems/classification.py",
            "problems/conditional.py",
            "catalog/query.py",
            "catalog/experiment/query.py",
            "pipeline/base.py",
            "catalog/registry.py",
            "catalog/dashboard.py",
            "catalog/experiment/dashboard.py",
            "bias/base.py",
        ),
    )
    _require_dirs(
        package_root,
        diags,
        (
            "adapters",
            "representations",
            "problems",
            "capabilities",
            "pipeline",
            "assembly",
            "assembly/config",
            "assembly/schema",
            "problems/training",
            "pipeline/numericizer",
            "pipeline/conditional",
            "representations/heads",
            "catalog",
            "catalog/experiment",
            "bias",
        ),
    )
    _check_text_contract(package_root / "core" / "trainer.py", diags, required=("set_adapter", "evaluate_individual", "evaluate_population", "write_population_snapshot", "set_resource_context"))
    _check_text_contract(package_root / "core" / "resources.py", diags, required=("ResourceContext", "ResourceAudit", "coerce_resource_context"))
    _check_text_contract(package_root / "assembly" / "builders.py", diags, required=("build_trainer", "build_pipeline"))
    _check_text_contract(package_root / "problems" / "proxy.py", diags, required=("MLBlackTrainingProxy", "evaluate_population", "TrainingResultRecord"))
    _check_text_contract(package_root / "pipeline" / "numericizer" / "plan.py", diags, required=("NumericizationPlan", "NumericFeatureColumn"))
    _check_text_contract(package_root / "representations" / "heads" / "probability.py", diags, required=("BinaryLogisticHead", "SoftmaxHead", "ProbabilityCalibrationHead"))
    _check_text_contract(package_root / "problems" / "classification.py", diags, required=("auc_roc", "average_precision", "f1"))
    _check_context_contracts(package_root, diags, strict=strict)
    _check_standard_case_scaffolds(package_root, diags)

    if strict:
        _check_no_large_context_antipatterns(package_root, diags)

    diags.append(
        DoctorDiagnostic(
            "info",
            "doctor-scope",
            "Checked mlblack ML-specialized scaffold boundaries: core, passive resource context, single-trainer assembly, pipeline, catalog.",
            str(package_root),
        )
    )
    return DoctorReport(project_root=package_root, diagnostics=tuple(diags))


def format_doctor_report(report: DoctorReport) -> str:
    lines = [f"mlblack doctor: {'ok' if report.ok else 'issues'}", f"root: {report.project_root}"]
    for item in report.diagnostics:
        suffix = "" if not item.path else f" ({item.path})"
        lines.append(f"[{item.level}] {item.code}: {item.message}{suffix}")
    return "\n".join(lines)


def iter_diagnostics_by_level(diagnostics: Iterable[DoctorDiagnostic], level: str) -> list[DoctorDiagnostic]:
    target = str(level)
    return [item for item in diagnostics if item.level == target]


def _require_files(root: Path, diags: list[DoctorDiagnostic], rel_paths: tuple[str, ...]) -> None:
    for rel in rel_paths:
        path = root / rel
        if not path.is_file():
            diags.append(DoctorDiagnostic("error", "missing-file", f"Required file is missing: {rel}", str(path)))


def _require_dirs(root: Path, diags: list[DoctorDiagnostic], rel_paths: tuple[str, ...]) -> None:
    for rel in rel_paths:
        path = root / rel
        if not path.is_dir():
            diags.append(DoctorDiagnostic("error", "missing-dir", f"Required directory is missing: {rel}", str(path)))


def _check_text_contract(path: Path, diags: list[DoctorDiagnostic], *, required: tuple[str, ...]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    for token in required:
        if token not in text:
            diags.append(DoctorDiagnostic("error", "missing-contract-token", f"Expected token not found: {token}", str(path)))


def _check_no_large_context_antipatterns(root: Path, diags: list[DoctorDiagnostic]) -> None:
    for path in root.rglob("*.py"):
        if any(part in {".conda", ".git", ".pytest_cache", "__pycache__", "site-packages"} for part in path.parts):
            continue
        if path.name == "doctor.py" and "project" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for token in ('context["population"]', "context['population']", 'context["history"]', "context['history']"):
            if token in text:
                diags.append(DoctorDiagnostic("warn", "large-context-write", f"Potential large object context write: {token}", str(path)))


_CASE_MARKER_FILES = {
    "build_solver.py",
    "build_trainer.py",
    "run_solver.py",
    "run_trainer.py",
    "project_registry.py",
}
_CASE_MARKER_DIRS = {
    "problem",
    "pipeline",
    "adapter",
    "plugins",
    "solver",
    "runtime",
    "evaluation",
    "bias",
}
_NON_CASE_DIR_NAMES = {
    "problem",
    "pipeline",
    "adapter",
    "plugins",
    "solver",
    "runtime",
    "evaluation",
    "bias",
    "assembly",
    "catalog",
    "config",
    "original",
    "assets",
    "docs",
}


def _check_standard_case_scaffolds(root: Path, diags: list[DoctorDiagnostic]) -> None:
    examples_cases = root / "examples" / "cases"
    if not examples_cases.is_dir():
        return

    case_roots = tuple(_iter_case_roots(examples_cases))
    for case_root in case_roots:
        _check_case_root_scaffold(case_root, diags)
    diags.append(
        DoctorDiagnostic(
            "info",
            "case-scaffold-scope",
            f"Validated {len(case_roots)} examples/cases standard scaffold roots.",
            str(examples_cases),
        )
    )


def _iter_case_roots(examples_cases: Path) -> Iterable[Path]:
    candidates = [examples_cases]
    candidates.extend(path for path in examples_cases.rglob("*") if path.is_dir())
    for directory in sorted(candidates):
        if directory == examples_cases:
            continue
        rel_parts = directory.relative_to(examples_cases).parts
        if any(part in _NON_CASE_DIR_NAMES or part == "__pycache__" for part in rel_parts):
            continue
        child_files = {path.name for path in directory.iterdir() if path.is_file()}
        child_dirs = {path.name for path in directory.iterdir() if path.is_dir()}
        if child_files & _CASE_MARKER_FILES or child_dirs & _CASE_MARKER_DIRS:
            yield directory


def _check_case_root_scaffold(case_root: Path, diags: list[DoctorDiagnostic]) -> None:
    build_solver = case_root / "build_solver.py"
    build_trainer = case_root / "build_trainer.py"
    run_solver = case_root / "run_solver.py"
    run_trainer = case_root / "run_trainer.py"

    if not build_solver.is_file():
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-missing-build-solver",
                "Case must use build_solver.py as canonical assembly entry.",
                str(build_solver),
            )
        )
    elif "def build_solver" not in _read_text(build_solver):
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-build-solver-missing-function",
                "build_solver.py must define build_solver().",
                str(build_solver),
            )
        )

    if build_trainer.is_file():
        if not build_solver.is_file():
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "case-build-trainer-without-build-solver",
                    "build_trainer.py is only an alias and requires build_solver.py.",
                    str(build_trainer),
                )
            )
        _check_alias_file(
            build_trainer,
            diags,
            required_tokens=("build_solver", "build_trainer"),
            forbidden_tokens=("def build_trainer", "def build_project_trainer"),
            code="case-build-trainer-not-alias",
            message="build_trainer.py must be a thin alias to build_solver.build_solver.",
        )

    if not run_solver.is_file():
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-missing-run-solver",
                "Case must use run_solver.py as canonical CLI entry.",
                str(run_solver),
            )
        )

    if run_trainer.is_file():
        if not run_solver.is_file():
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "case-run-trainer-without-run-solver",
                    "run_trainer.py is only an alias and requires run_solver.py.",
                    str(run_trainer),
                )
            )
        _check_alias_file(
            run_trainer,
            diags,
            required_tokens=("run_solver", "main"),
            forbidden_tokens=("build_trainer", "def main("),
            code="case-run-trainer-not-alias",
            message="run_trainer.py must be a thin alias to run_solver.main.",
        )

    legacy_capabilities = case_root / "capabilities"
    if legacy_capabilities.is_dir():
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-legacy-capabilities-dir",
                "Case-level capabilities/ is forbidden; use plugins/.",
                str(legacy_capabilities),
            )
        )

    legacy_representation = case_root / "representation"
    if legacy_representation.is_dir():
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-legacy-representation-dir",
                "Case-level representation/ is forbidden; use pipeline/representation/.",
                str(legacy_representation),
            )
        )

    legacy_scaffold_json = case_root / "assembly" / "scaffold.json"
    if legacy_scaffold_json.is_file():
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-legacy-assembly-scaffold-json",
                "assembly/scaffold.json is forbidden; assembly logic belongs in build_solver.py.",
                str(legacy_scaffold_json),
            )
        )


def _check_alias_file(
    path: Path,
    diags: list[DoctorDiagnostic],
    *,
    required_tokens: tuple[str, ...],
    forbidden_tokens: tuple[str, ...],
    code: str,
    message: str,
) -> None:
    text = _read_text(path)
    if not all(token in text for token in required_tokens):
        diags.append(DoctorDiagnostic("error", code, message, str(path)))
        return
    for token in forbidden_tokens:
        if token in text:
            diags.append(DoctorDiagnostic("error", code, message, str(path)))
            return


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="replace")


def _check_context_contracts(root: Path, diags: list[DoctorDiagnostic], *, strict: bool) -> None:
    parent = root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    seen: set[str] = set()
    component_count = 0
    for module_name, path in _iter_python_modules(root):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            level = "error" if strict else "warn"
            diags.append(DoctorDiagnostic(level, "component-import-failed", f"Could not import {module_name}: {exc!r}", str(path)))
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, "__module__", "") != module_name:
                continue
            key = f"{obj.__module__}:{obj.__name__}"
            if key in seen:
                continue
            seen.add(key)
            if not _is_contract_component(obj):
                continue
            component_count += 1
            _check_component_contract(obj, diags, strict=strict, path=path)
    diags.append(
        DoctorDiagnostic(
            "info",
            "context-contract-scope",
            f"Validated {component_count} nsgablack-style context component contracts.",
            str(root),
        )
    )


def _iter_python_modules(root: Path) -> Iterable[tuple[str, Path]]:
    package_name = root.name
    include_top_level = {
        "adapters",
        "assembly",
        "backends",
        "bias",
        "capabilities",
        "catalog",
        "core",
        "integrations",
        "models",
        "pipeline",
        "presets",
        "problems",
        "project",
        "representations",
    }
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] not in include_top_level and path.name not in {"__init__.py", "mlblack.py"}:
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(part for part in (package_name, *parts) if part)
        yield module_name, path


def _is_contract_component(obj: Any) -> bool:
    attrs = (
        "context_requires",
        "context_optional",
        "context_provides",
        "context_mutates",
        "context_cache",
        "requires_metrics",
        "metrics_fallback",
        "context_notes",
    )
    namespace = getattr(obj, "__dict__", {})
    has_context_attr = any(attr in namespace for attr in attrs)
    raw_contract = namespace.get("contract")
    has_contract = isinstance(raw_contract, (ComponentContract, ContextContract, dict))
    return bool(has_context_attr or has_contract)


def _check_component_contract(obj: Any, diags: list[DoctorDiagnostic], *, strict: bool, path: Path) -> None:
    level = "error" if strict else "warn"
    namespace = getattr(obj, "__dict__", {})
    raw_contract = namespace.get("contract")
    has_explicit_context = any(
        attr in namespace
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
    if raw_contract is not None and not has_explicit_context:
        diags.append(
            DoctorDiagnostic(
                "warn",
                "legacy-contract-only",
                f"{obj.__module__}:{obj.__name__} declares contract but no context_* class attributes.",
                str(path),
            )
        )
    try:
        contract = ContextContract.from_component(obj, fallback_contract=raw_contract)
    except Exception as exc:
        diags.append(DoctorDiagnostic(level, "invalid-context-contract", f"{obj.__module__}:{obj.__name__}: {exc!r}", str(path)))
        return
    unknown = contract.unknown_keys()
    if unknown:
        diags.append(
            DoctorDiagnostic(
                level,
                "unknown-context-key",
                f"{obj.__module__}:{obj.__name__} declares unknown context keys: {unknown}",
                str(path),
            )
        )
    unknown_metrics = contract.unknown_metric_keys()
    if unknown_metrics:
        diags.append(
            DoctorDiagnostic(
                level,
                "unknown-metric-key",
                f"{obj.__module__}:{obj.__name__} declares unknown requires_metrics keys: {unknown_metrics}; known={METRIC_KEYS}",
                str(path),
            )
        )
    if contract.metrics_fallback not in METRIC_FALLBACKS:
        diags.append(
            DoctorDiagnostic(
                level,
                "invalid-metrics-fallback",
                f"{obj.__module__}:{obj.__name__} declares invalid metrics_fallback: {contract.metrics_fallback}",
                str(path),
            )
        )

