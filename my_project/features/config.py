from __future__ import annotations

from my_project.config.schema import FeatureConfig
from my_project.features.example_feature_builder import FeatureBundle, build_features
from my_project.problem.example_problem import ProblemContext


def build_feature_bundle(problem: ProblemContext, cfg: FeatureConfig) -> FeatureBundle:
    return build_features(problem, add_bias=bool(cfg.add_bias))
