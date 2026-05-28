# -*- coding: utf-8 -*-
# Project registry: register custom components for catalog discovery.

from __future__ import annotations

from mlblack.catalog.registry import get_catalog


def register_project_components():
    catalog = get_catalog()
    # Example:
    # catalog.register("problem.my_project_regression", {
    #     "title": "My Project Regression Problem",
    #     "kind": "problem",
    #     "import_path": "problem.example_problem:ExampleRegressionProblem",
    #     "tags": ["example"],
    #     "summary": "Custom regression problem for this project.",
    # })
    pass


if __name__ == "__main__":
    register_project_components()
