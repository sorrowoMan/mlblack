from __future__ import annotations

from my_project.known_relation_symbolic.pipeline.bundle import build_known_relation_bundle


def build_known_relation_train_bundle(**kwargs):
    return build_known_relation_bundle(**kwargs)


__all__ = ["build_known_relation_bundle", "build_known_relation_train_bundle"]
