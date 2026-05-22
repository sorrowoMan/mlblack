from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mlblack.assembly.config import default_scaffold_config, merge_config


def create_standard_scaffold(
    path: str | Path,
    *,
    name: str = "mlblack_project",
    features: tuple[str, ...] = ("x0",),
    target: str = "target",
    config_overrides: Mapping[str, Any] | None = None,
    exist_ok: bool = True,
) -> dict[str, Path]:
    """Create a minimal single-trainer ML scaffold.

    Multi-stage/group/event orchestration belongs to nsgablack; generated
    mlblack projects expose only an inner trainer builder.
    """

    root = Path(path)
    if root.exists() and not exist_ok:
        raise FileExistsError(str(root))
    root.mkdir(parents=True, exist_ok=True)
    for rel in ("config", "data", "reports", "artifacts"):
        (root / rel).mkdir(parents=True, exist_ok=True)

    config = merge_config(default_scaffold_config(name=name, features=features, target=target), config_overrides)
    config_path = root / "config" / "scaffold.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    build_path = root / "build_trainer.py"
    build_path.write_text(
        "from pathlib import Path\n"
        "from mlblack.assembly.schema import load_scaffold_config\n"
        "from mlblack.assembly import build_pipeline, build_trainer\n\n"
        "def build_project_trainer(data, config_path=None):\n"
        "    path = Path(config_path or Path(__file__).parent / 'config' / 'scaffold.json')\n"
        "    config = load_scaffold_config(path)\n"
        "    inner_training = dict(config.inner_training)\n"
        "    pipeline = build_pipeline(inner_training.get('pipeline'))\n"
        "    resource_context = dict(inner_training.get('resource_context', {}) or {})\n"
        "    prepared = pipeline.fit_transform(data, resource_context)\n"
        "    trainer_spec = dict(inner_training.get('trainer', {}) or {})\n"
        "    if resource_context and not trainer_spec.get('resource_context'):\n"
        "        trainer_spec['resource_context'] = resource_context\n"
        "    trainer = build_trainer(trainer_spec, prepared)\n"
        "    trainer.context_store['pipeline'] = pipeline.describe()\n"
        "    return trainer\n",
        encoding="utf-8",
    )

    run_path = root / "run_trainer.py"
    run_path.write_text(
        "import numpy as np\n"
        "from mlblack.pipeline.data import train_valid_split\n"
        "from build_trainer import build_project_trainer\n\n"
        "if __name__ == '__main__':\n"
        "    X = np.linspace(-1, 1, 64).reshape(-1, 1)\n"
        "    y = 1.5 + 2.0 * X[:, 0]\n"
        "    data = train_valid_split(X, y, feature_names=('x0',))\n"
        "    trainer = build_project_trainer(data)\n"
        "    result = trainer.fit(max_steps=20)\n"
        "    print(trainer.build_report())\n"
        "    print(result.report)\n",
        encoding="utf-8",
    )

    return {"root": root, "config": config_path, "build_trainer": build_path, "run_trainer": run_path}

