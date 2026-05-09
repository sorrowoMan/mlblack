from __future__ import annotations

from my_project.config.schema import ModelConfig
from my_project.features.example_feature_builder import FeatureBundle
from my_project.model.example_model import ModelResult, train_and_predict


def train_model_bundle(bundle: FeatureBundle, cfg: ModelConfig) -> ModelResult:
    return train_and_predict(bundle, baseline=str(cfg.baseline))
