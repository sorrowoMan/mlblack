from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from .run_case import run_case
except ImportError:  # direct script execution
    from run_case import run_case


@dataclass
class TinyTransformerSmokeRunner:
    steps: int = 1
    output_dir: Path = Path("runs/tiny_transformer_smoke")
    resource_context: object | None = None

    def set_resource_context(self, context):
        self.resource_context = context
        return self

    def run(self):
        return run_case(steps=self.steps, output_dir=self.output_dir)


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    runner = TinyTransformerSmokeRunner()
    if resource_context is not None:
        runner.set_resource_context(resource_context)
    return runner
