from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from blackbase.project.doctor import (
    DoctorDiagnostic,
    DoctorReport,
    run_common_project_doctor as _run_common_project_doctor,
)

from blackbase.context import ContextContract
from blackbase.context import METRIC_FALLBACKS, METRIC_KEYS, unknown_context_keys
from blackbase.contracts import ComponentContract

def run_project_doctor(path: str | Path | None = None, *, strict: bool = False) -> DoctorReport:
    root = Path(path or Path.cwd()).resolve()
    package_root = root / "mlblack" if (root / "mlblack").is_dir() else root
    diags: list[DoctorDiagnostic] = []

    common_report = _run_common_project_doctor(package_root, strict=bool(strict))
    for item in common_report.diagnostics:
        diags.append(
            DoctorDiagnostic(
                str(item.level),
                str(item.code),
                str(item.message),
                str(item.path or ""),
            )
        )

    if _is_user_project_root(package_root):
        diags.append(
            DoctorDiagnostic(
                "info",
                "doctor-scope",
                "Checked user Project root through the shared blackbase Project/Case/Scaffold rules.",
                str(package_root),
            )
        )
        return DoctorReport(project_root=package_root, diagnostics=tuple(diags))

    _require_files(
        package_root,
        diags,
        (
            "core/representation.py",
            "core/problem.py",
            "core/state.py",
            "core/artifacts.py",
            "integrations/nsgablack_control.py",
            "integrations/nsgablack_optimization.py",
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
            "catalog/entries",
            "catalog/experiment",
            "bias",
        ),
    )
    _check_text_contract(
        package_root / "integrations" / "nsgablack_control.py",
        diags,
        required=("LearningSolver", "ComposableSolver", "fit", "set_resource_context"),
    )
    _check_text_contract(package_root / "assembly" / "builders.py", diags, required=("build_trainer", "build_pipeline"))
    _check_text_contract(package_root / "problems" / "proxy.py", diags, required=("MLBlackTrainingProxy", "evaluate_population", "TrainingResultRecord"))
    _check_text_contract(package_root / "pipeline" / "numericizer" / "plan.py", diags, required=("NumericizationPlan", "NumericFeatureColumn"))
    _check_text_contract(package_root / "representations" / "heads" / "probability.py", diags, required=("BinaryLogisticHead", "SoftmaxHead", "ProbabilityCalibrationHead"))
    _check_text_contract(package_root / "problems" / "classification.py", diags, required=("auc_roc", "average_precision", "f1"))
    _check_context_contracts(package_root, diags, strict=strict)
    _check_project_wrappers(package_root, diags)
    _check_standard_case_scaffolds(package_root, diags)

    if strict:
        _check_no_large_context_antipatterns(package_root, diags)

    diags.append(
        DoctorDiagnostic(
            "info",
            "doctor-scope",
            "Checked MLBlack semantic-extension boundaries: ML Problem/Representation/Provider, NSGABlack control integration, passive resource context, pipeline, and catalog.",
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


def _is_user_project_root(root: Path) -> bool:
    return (
        (root / "project_config.py").is_file()
        and (root / "run_project.py").is_file()
        and (root / "cases").is_dir()
        and not (root / "core" / "trainer.py").is_file()
    )


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
_PROJECT_REQUIRED_FILES = ("README.md", "project_config.py", "run_project.py")
_LEGACY_CASE_DOCS = ("BUILD_SOLVER_REGISTRATION.md", "COMPONENT_REGISTRATION.md")
_LEGACY_CASE_DIRS_WARN = ("assembly", "case_scaffold")


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


def _check_project_wrappers(root: Path, diags: list[DoctorDiagnostic]) -> None:
    examples_cases = root / "examples" / "cases"
    if not examples_cases.is_dir():
        return

    projects = tuple(path for path in sorted(examples_cases.iterdir()) if _is_project_root(path))
    for project_root in projects:
        _check_project_root(project_root, diags)
    diags.append(
        DoctorDiagnostic(
            "info",
            "project-wrapper-scope",
            f"Validated {len(projects)} examples/cases Project wrapper roots.",
            str(examples_cases),
        )
    )


def _is_project_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    child_files = {child.name for child in path.iterdir() if child.is_file()}
    child_dirs = {child.name for child in path.iterdir() if child.is_dir()}
    return {"project_config.py", "run_project.py"}.issubset(child_files) and "cases" in child_dirs


def _check_project_root(project_root: Path, diags: list[DoctorDiagnostic]) -> None:
    for filename in _PROJECT_REQUIRED_FILES:
        path = project_root / filename
        if not path.is_file():
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-missing-required-file",
                    f"Project wrapper must include {filename}.",
                    str(path),
                )
            )

    cases_dir = project_root / "cases"
    if not cases_dir.is_dir():
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-missing-cases-dir",
                "Project wrapper must include cases/.",
                str(cases_dir),
            )
        )
    elif not (cases_dir / "__init__.py").is_file():
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-cases-missing-init",
                "Project cases/ must include __init__.py for stable Case imports.",
                str(cases_dir / "__init__.py"),
            )
        )

    config_path = project_root / "project_config.py"
    if config_path.is_file():
        text = _read_text(config_path)
        if "STAGES" not in text or "GROUPS" not in text:
            diags.append(
                DoctorDiagnostic(
                    "warn",
                    "project-config-missing-stage-group",
                    "project_config.py should declare STAGES and GROUPS for explicit Project orchestration.",
                    str(config_path),
                )
            )
        if "L0" not in text:
            diags.append(
                DoctorDiagnostic(
                    "warn",
                    "project-config-missing-l0",
                    "project_config.py should declare L0 so ResourceContext grants are auditable.",
                    str(config_path),
                )
            )

    run_project = project_root / "run_project.py"
    if run_project.is_file():
        text = _read_text(run_project)
        if "mlblack.project.project_runner" not in text and "run_project" not in text:
            diags.append(
                DoctorDiagnostic(
                    "warn",
                    "project-runner-nonstandard",
                    "run_project.py should delegate to the shared Project/Case runner or expose an equivalent substrate surface.",
                    str(run_project),
                )
            )

    if cases_dir.is_dir():
        for child in sorted(cases_dir.iterdir()):
            if child.is_dir() and child.name != "__pycache__" and not (child / "__init__.py").is_file():
                diags.append(
                    DoctorDiagnostic(
                        "error",
                        "case-package-missing-init",
                        "Each cases/<case_name>/ directory must include __init__.py.",
                        str(child / "__init__.py"),
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
        if {"project_config.py", "run_project.py"}.issubset(child_files) and "cases" in child_dirs:
            continue
        if child_files & _CASE_MARKER_FILES or child_dirs & _CASE_MARKER_DIRS:
            yield directory


def _check_case_root_scaffold(case_root: Path, diags: list[DoctorDiagnostic]) -> None:
    build_solver = case_root / "build_solver.py"
    build_trainer = case_root / "build_trainer.py"
    run_solver = case_root / "run_solver.py"
    run_trainer = case_root / "run_trainer.py"

    build_solver_text = ""
    if not build_solver.is_file():
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-missing-build-solver",
                "Case must use build_solver.py as canonical assembly entry.",
                str(build_solver),
            )
        )
    else:
        build_solver_text = _read_text(build_solver)
        if "def build_solver" not in build_solver_text:
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "case-build-solver-missing-function",
                    "build_solver.py must define build_solver().",
                    str(build_solver),
                )
            )
        else:
            _check_build_entry_signature(build_solver, build_solver_text, entry_name="build_solver", diags=diags)

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

    pipeline_main = case_root / "pipeline" / "main.py"
    pipeline_module = case_root / "pipeline.py"
    if not pipeline_main.is_file() and not pipeline_module.is_file():
        diags.append(
            DoctorDiagnostic(
                "warn",
                "case-pipeline-entry-recommended",
                "Recommended: add one canonical pipeline entry (pipeline/main.py or pipeline.py) and compose operators inside it.",
                str(case_root / "pipeline"),
            )
        )

    for dirname in _LEGACY_CASE_DIRS_WARN:
        legacy_dir = case_root / dirname
        if legacy_dir.is_dir():
            diags.append(
                DoctorDiagnostic(
                    "warn",
                    "case-legacy-scaffold-dir",
                    f"{dirname}/ is compatibility/internal helper surface only; canonical assembly belongs in build_solver.py and standard Case directories.",
                    str(legacy_dir),
                )
            )

    for filename in _LEGACY_CASE_DOCS:
        doc_path = case_root / filename
        if doc_path.is_file():
            diags.append(
                DoctorDiagnostic(
                    "warn",
                    "case-legacy-registration-doc",
                    f"{filename} is legacy registration guidance; prefer the Case README and project-level scaffold docs.",
                    str(doc_path),
                )
            )

    runtime_config = case_root / "runtime" / "config.py"
    if runtime_config.is_file():
        text = _read_text(runtime_config)
        if "Project L0 runtime" in text or "Project-level L0" in text:
            diags.append(
                DoctorDiagnostic(
                    "warn",
                    "case-runtime-project-l0-wording",
                    "Case runtime/config.py should describe requirement/profile/audit only; Project L0 grants resources at the Project root.",
                    str(runtime_config),
                )
            )


def _check_build_entry_signature(path: Path, text: str, *, entry_name: str, diags: list[DoctorDiagnostic]) -> None:
    try:
        import ast

        tree = ast.parse(text)
    except SyntaxError as exc:
        diags.append(DoctorDiagnostic("error", "case-build-solver-syntax-error", f"Could not parse build_solver.py: {exc}", str(path)))
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_name:
            arg_names = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            if "resource_context" not in arg_names:
                diags.append(
                    DoctorDiagnostic(
                        "warn",
                        "case-build-entry-missing-resource-context",
                        f"{entry_name}() should accept resource_context so Project L0 grants can be injected and audited.",
                        str(path),
                    )
                )
            if "component_overrides" not in arg_names:
                diags.append(
                    DoctorDiagnostic(
                        "warn",
                        "case-build-entry-missing-component-overrides",
                        f"{entry_name}() should accept component_overrides for nested and cross-framework Case composition.",
                        str(path),
                    )
                )
            return


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
    excluded_top_level = {"examples"}
    for path in root.rglob("*.py"):
        if any(part in {".conda", ".git", ".pytest_cache", ".mypy_cache", "__pycache__", "site-packages", "runs"} for part in path.parts):
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in excluded_top_level:
            continue
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
    normalized = contract.normalized()
    unknown = tuple(unknown_context_keys(normalized.all_context_keys()))
    if unknown:
        diags.append(
            DoctorDiagnostic(
                level,
                "unknown-context-key",
                f"{obj.__module__}:{obj.__name__} declares unknown context keys: {unknown}",
                str(path),
            )
        )
    requires_metrics = tuple(str(item) for item in getattr(obj, "requires_metrics", getattr(raw_contract, "requires_metrics", ())) or ())
    unknown_metrics = tuple(item for item in requires_metrics if item not in METRIC_KEYS)
    if unknown_metrics:
        diags.append(
            DoctorDiagnostic(
                level,
                "unknown-metric-key",
                f"{obj.__module__}:{obj.__name__} declares unknown requires_metrics keys: {unknown_metrics}; known={METRIC_KEYS}",
                str(path),
            )
        )
    metrics_fallback = str(getattr(obj, "metrics_fallback", getattr(raw_contract, "metrics_fallback", "strict")) or "strict")
    if metrics_fallback not in METRIC_FALLBACKS:
        diags.append(
            DoctorDiagnostic(
                level,
                "invalid-metrics-fallback",
                f"{obj.__module__}:{obj.__name__} declares invalid metrics_fallback: {metrics_fallback}",
                str(path),
            )
        )

