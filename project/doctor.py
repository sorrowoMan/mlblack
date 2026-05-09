from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from project.doctor_rules import register_builtin_rules
from project.doctor_types import DoctorProblem, DoctorRule


def _is_error(problem: DoctorProblem) -> bool:
    return str(problem.severity).strip().lower() == "error"


def _discover_external_rules(rules_dir: Path) -> list[DoctorRule]:
    out: list[DoctorRule] = []
    if not rules_dir.exists() or not rules_dir.is_dir():
        return out

    for py in sorted(rules_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        mod_name = f"mlblack_doctor_ext_{py.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, py)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

        register_fn = getattr(module, "register_rules", None)
        if callable(register_fn):
            rules = register_fn()
            for item in rules:
                if isinstance(item, DoctorRule):
                    out.append(item)
    return out


def _build_rule_set(*, rules_dir: str | Path | None = None) -> list[DoctorRule]:
    rules = list(register_builtin_rules())
    if rules_dir is not None:
        rules.extend(_discover_external_rules(Path(rules_dir).resolve()))

    # keep first-seen rule id; ignore duplicates
    uniq: list[DoctorRule] = []
    seen: set[str] = set()
    for rule in rules:
        rid = str(rule.rule_id).strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        uniq.append(rule)
    return uniq


def _select_rules(
    rules: list[DoctorRule],
    *,
    only_rule_ids: list[str] | None,
) -> list[DoctorRule]:
    if not only_rule_ids:
        return list(rules)
    wanted = {str(x).strip() for x in only_rule_ids if str(x).strip()}
    return [r for r in rules if str(r.rule_id) in wanted]


def run_doctor(
    root: str | Path = ".",
    *,
    rules_dir: str | Path | None = None,
    only_rule_ids: list[str] | None = None,
) -> list[DoctorProblem]:
    base = Path(root).resolve()
    rules = _select_rules(_build_rule_set(rules_dir=rules_dir), only_rule_ids=only_rule_ids)

    problems: list[DoctorProblem] = []
    for rule in rules:
        try:
            items = list(rule.run(base))
            for p in items:
                if isinstance(p, DoctorProblem):
                    problems.append(p)
        except Exception as exc:
            problems.append(
                DoctorProblem(
                    severity="error",
                    code="rule_runtime_error",
                    message=f"rule '{rule.rule_id}' failed: {type(exc).__name__}: {exc}",
                )
            )
    return problems


def _print_problem_lines(items: list[DoctorProblem]) -> None:
    if not items:
        print("OK doctor found no problems")
        return
    for x in items:
        p = f" path={x.path}" if x.path else ""
        print(f"{x.severity.upper()} {x.code} {x.message}{p}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mlblack project doctor")
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--format", type=str, default="problem", choices=["problem", "json"])
    parser.add_argument(
        "--rules-dir",
        type=str,
        default=None,
        help="Optional directory for external doctor rule plugins (*.py with register_rules()).",
    )
    parser.add_argument(
        "--only-rule",
        action="append",
        default=None,
        help="Run only selected rule id (repeatable).",
    )
    args = parser.parse_args(argv)

    items = run_doctor(
        args.path,
        rules_dir=args.rules_dir,
        only_rule_ids=args.only_rule,
    )

    if args.format == "json":
        print(json.dumps([x.to_dict() for x in items], ensure_ascii=False, indent=2))
    else:
        _print_problem_lines(items)

    if bool(args.strict) and any(_is_error(x) for x in items):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
