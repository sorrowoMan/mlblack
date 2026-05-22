from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .spec import DatasetSchema, ScaffoldConfig


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_dataset_schema(path: str | Path) -> DatasetSchema:
    return DatasetSchema.from_value(load_json(path))


def load_scaffold_config(path: str | Path) -> ScaffoldConfig:
    return ScaffoldConfig.from_value(load_json(path))


def save_scaffold_config(config: ScaffoldConfig | Mapping[str, Any], path: str | Path) -> Path:
    return dump_json(ScaffoldConfig.from_value(config).as_dict(), path)
