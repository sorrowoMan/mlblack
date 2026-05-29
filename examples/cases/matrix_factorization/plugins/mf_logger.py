# -*- coding: utf-8 -*-
"""Per-step metrics logger plugin for matrix factorization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mlblack.core.plugin import Plugin


class MetricsLoggerPlugin(Plugin):
    """Logs per-step metrics to a JSON lines file."""

    context_requires = ()
    context_provides = ("plugin.side_effect",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Writes step metrics to a log file."

    def __init__(self, log_dir="logs", *, name="metrics_logger"):
        self.name = name
        self.log_dir = Path(log_dir)
        self._log_path = None

    def on_fit_start(self, trainer, context):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        run_name = context.get("run_name", "mf_run")
        self._log_path = self.log_dir / f"{run_name}_steps.jsonl"

    def on_step_end(self, trainer, context, row):
        if self._log_path is None:
            return
        payload = dict(row)
        line = json.dumps(payload, default=str)
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
