from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__


def _cmd_catalog(argv: Sequence[str]) -> int:
    from catalog.cli import main as catalog_main

    return int(catalog_main(list(argv)))


def _cmd_doctor(argv: Sequence[str]) -> int:
    from project.doctor import main as doctor_main

    return int(doctor_main(list(argv)))


def _cmd_experiment(argv: Sequence[str]) -> int:
    from experiment.cli import main as experiment_main

    return int(experiment_main(list(argv)))


def _cmd_scaffold_init(args: argparse.Namespace) -> int:
    from project import init_project

    root = init_project(Path(args.path), force=bool(args.force))
    print(f"initialized scaffold: {root}")
    print("next:")
    print(f"  cd {root}")
    print("  python run_train.py --config configs/train_config.json")
    return 0


def _cmd_scaffold_run(args: argparse.Namespace) -> int:
    from project import load_scaffold_spec, run_project_scaffold

    try:
        from examples.path_defaults import apply_env_defaults

        apply_env_defaults()
    except Exception:
        # Keep CLI independent when examples package is unavailable.
        pass

    spec = load_scaffold_spec(args.config)
    result = run_project_scaffold(spec)

    print("SCAFFOLD RUN RESULT")
    print(f"trainer={result.report['trainer_name']}")
    print(f"artifact_id={result.report['artifact']['artifact_id']}")
    for split, metrics in result.metrics.items():
        print(
            f"{split:5s} rmse={metrics['rmse']:.4f}  mae={metrics['mae']:.4f}  r2={metrics['r2']:.4f}"
        )
    print(f"output_dir={result.output_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mlblack unified CLI")
    parser.add_argument("--version", action="version", version=f"mlblack {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("catalog", help="Catalog tools (pass-through to `python -m catalog.cli ...`)")
    sub.add_parser("doctor", help="Project doctor (pass-through to `python -m project.doctor ...`)")
    sub.add_parser("experiment", help="Experiment result surface (summary/list/show/ui)")

    p_scaffold = sub.add_parser("scaffold", help="Scaffold project operations")
    sub_scaffold = p_scaffold.add_subparsers(dest="scaffold_cmd", required=True)

    p_init = sub_scaffold.add_parser("init", help="Initialize a scaffold project directory")
    p_init.add_argument("--path", required=True, help="Target project directory")
    p_init.add_argument("--force", action="store_true", help="Allow non-empty directory")
    p_init.set_defaults(_fn=_cmd_scaffold_init)

    p_run = sub_scaffold.add_parser("run", help="Run one scaffold config")
    p_run.add_argument("--config", required=True, help="Path to scaffold config json")
    p_run.set_defaults(_fn=_cmd_scaffold_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw and raw[0] == "catalog":
        return _cmd_catalog(raw[1:])
    if raw and raw[0] == "doctor":
        return _cmd_doctor(raw[1:])
    if raw and raw[0] == "experiment":
        return _cmd_experiment(raw[1:])

    parser = _build_parser()
    args = parser.parse_args(raw)
    return int(args._fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
