# -*- coding: utf-8 -*-
# Project registry: register custom components for catalog discovery.

from __future__ import annotations

from mlblack.catalog.registry import get_catalog


def register_project_components():
    catalog = get_catalog()

    catalog.register("problem.matrix_factorization", {
        "title": "Matrix Factorization Regression Problem",
        "kind": "problem",
        "import_path": "problem.matrix_factorization_problem:MatrixFactorizationProblem",
        "tags": ["matrix_factorization", "recommendation", "collaborative_filtering"],
        "summary": "MSE reconstruction loss over observed (user, item) ratings with analytic gradients.",
    })

    catalog.register("representation.matrix_factorization", {
        "title": "Matrix Factorization U,V Embedding Representation",
        "kind": "representation",
        "import_path": "representation.mf_representation:MFRepresentation",
        "tags": ["matrix_factorization", "embedding", "low_rank"],
        "summary": "Encodes/decodes (U, V) embedding matrices. NMF via repair() projection.",
    })

    catalog.register("pipeline.mf_synthetic_data", {
        "title": "Synthetic Rating Matrix Generator",
        "kind": "pipeline",
        "import_path": "pipeline.mf_pipeline:generate_synthetic_ratings",
        "tags": ["matrix_factorization", "synthetic_data"],
        "summary": "Generates low-rank rating matrices with configurable sparsity and noise.",
    })

    catalog.register("case.matrix_factorization", {
        "title": "Matrix Factorization Case",
        "kind": "assembly",
        "import_path": "build_trainer:build_mf_trainer",
        "tags": ["matrix_factorization", "gradient_descent", "framework_adapter", "demo"],
        "summary": "Custom MF problem/representation wired to stable gradient.sgd.",
    })


if __name__ == "__main__":
    register_project_components()
