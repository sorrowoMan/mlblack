from pathlib import Path
from typing import Any, Optional

from mlblack.assembly.schema.parser import load_scaffold_config
from mlblack.assembly.builders import build_pipeline, build_trainer


def build_solver(data: Any = None, config_path: Optional[str] = None):
    """Canonical unified scaffold entry; returns the assembled Trainer."""

    return build_project_trainer(data=data, config_path=config_path)


def build_project_trainer(data: Any = None, config_path: Optional[str] = None):
    """Build a Trainer from the case scaffold config.

    Loads `config/scaffold.json` by default (or `config_path` if provided),
    builds the pipeline (if declared) to prepare `data`, then builds and
    returns the assembled Trainer instance.
    """
    cfg_path = Path(config_path) if config_path else Path(__file__).parent / "config" / "scaffold.json"
    config = load_scaffold_config(cfg_path)
    inner_training = dict(config.inner_training or {})

    pipeline_spec = inner_training.get("pipeline")
    pipeline = build_pipeline(pipeline_spec) if pipeline_spec else None

    resource_context = dict(inner_training.get("resource_context") or {})

    if pipeline is not None:
        prepared = pipeline.fit_transform(data, resource_context)
    else:
        prepared = data

    trainer_spec = dict(inner_training.get("trainer") or {})
    if resource_context and not trainer_spec.get("resource_context"):
        trainer_spec["resource_context"] = resource_context

    trainer = build_trainer(trainer_spec, prepared)

    try:
        if pipeline is not None and hasattr(trainer, "context_store"):
            trainer.context_store["pipeline"] = pipeline.describe()
    except Exception:
        # Ignore non-fatal issues when storing pipeline description
        pass

    return trainer
