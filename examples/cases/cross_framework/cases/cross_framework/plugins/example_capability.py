# -*- coding: utf-8 -*-
"""Example Plugin: checkpoint persistence."""

from __future__ import annotations

import json
from pathlib import Path

from mlblack.core.capability import Capability


class ExampleCheckpointPlugin(Capability):
    """Save trainer state to disk periodically."""

    context_requires = ()
    context_provides = ("plugin.checkpoint_path",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Writes trainer state as JSON every N steps."

    def __init__(self, checkpoint_dir="checkpoints", interval=10, *, name="checkpoint"):
        super().__init__(name=name)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval = max(1, int(interval))

    def on_fit_start(self, trainer, context):
        del trainer, context
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer, context, row):
        del context
        step = int(row.get("step", trainer.generation))
        if step % self.interval == 0:
            state = trainer.get_state()
            path = self.checkpoint_dir / f"step_{step:06d}.json"
            path.write_text(json.dumps(state, indent=2, default=str))
