#!/usr/bin/env python
"""
Command line entrypoint for nsgablack.

This CLI intentionally stays small:
- `catalog`: discoverability (where is X?)

Usage:
  python -m nsgablack catalog search vns
  python -m nsgablack catalog list --kind adapter
  python -m nsgablack catalog show adapter.vns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow running from anywhere
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """
    nsgablack command-line interface.
    Provides access to catalog, project management, and health checks.
    """
    parser = argparse.ArgumentParser(
        description="nsgablack command-line interface.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Catalog Command ---
    parser_catalog = subparsers.add_parser(
        "catalog",
        help="Browse the component catalog.",
        description="Search, list, and show components like adapters, plugins, etc.",
    )
    # This is a placeholder for the full catalog CLI implementation
    parser_catalog.add_argument("catalog_sub", nargs="?", default="list", help="E.g., list, search, show")
    parser_catalog.add_argument("--kind", type=str, help="Filter by component kind.")

    # --- Project Command ---
    parser_project = subparsers.add_parser(
        "project",
        help="Manage projects and cases.",
        description="Create new projects or add cases (solvers/trainers) to an existing project.",
    )
    project_subparsers = parser_project.add_subparsers(dest="project_command", required=True)

    # `project new`
    parser_new = project_subparsers.add_parser("new", help="Create a new project scaffold.")
    parser_new.add_argument("project_name", type=str, help="The name of the new project directory.")

    # `project add-case`
    parser_add = project_subparsers.add_parser("add-case", help="Add a new case to the current project.")
    parser_add.add_argument("case_name", type=str, help="The name of the new case directory.")
    parser_add.add_argument(
        "--type",
        type=str,
        choices=["solver", "trainer"],
        required=True,
        help="The type of the case ('solver' for nsgablack, 'trainer' for mlblack).",
    )

    # --- Doctor Command ---
    parser_doctor = subparsers.add_parser(
        "doctor",
        help="Run project health checks.",
        description="Diagnose issues with project structure, dependencies, and configuration.",
    )
    # This is a placeholder for the full doctor CLI implementation
    parser_doctor.add_argument("--path", type=str, default=".", help="Path to the project to inspect.")
    parser_doctor.add_argument("--strict", action="store_true", help="Fail on any issue found.")

    args, unknown = parser.parse_known_args()

    if args.command == "catalog":
        print("Catalog command is not fully implemented yet.")
        # Example of how it would be called:
        # from nsgablack.catalog.cli import run_catalog_command
        # run_catalog_command(args)

    elif args.command == "project":
        # The actual logic is in scaffold.py, imported here to keep __main__ clean.
        try:
            from nsgablack.project.scaffold import create_project, add_case

            if args.project_command == "new":
                create_project(args.project_name)
            elif args.project_command == "add-case":
                add_case(args.case_name, args.type)
        except ImportError:
            print("Error: Could not import project management functions. Check your installation.")
            sys.exit(1)

    elif args.command == "doctor":
        print("Doctor command is not fully implemented yet.")
        # Example of how it would be called:
        # from nsgablack.project.doctor import run_project_doctor, format_doctor_report
        # print(format_doctor_report(run_project_doctor(args.path, strict=args.strict)))

    else:
        # If no command is specified, print help
        parser.print_help()


if __name__ == "__main__":
    main()
