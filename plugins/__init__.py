from __future__ import annotations

from .config import ReportingConfig, emit_report

__all__ = [
    "ReportingConfig",
    "emit_report",
    "ReproducibilityPlugin",
    "write_summary_report",
    "ReportWriterPlugin",
    "RuntimeResourcePlugin",
    "TrainerStateCheckpointPlugin",
]


def __getattr__(name: str):
    key = str(name)
    if key == "ReproducibilityPlugin":
        from .reproducibility_plugin import ReproducibilityPlugin

        return ReproducibilityPlugin
    if key == "write_summary_report":
        from .report_writer import write_summary_report

        return write_summary_report
    if key == "ReportWriterPlugin":
        from .report_writer_plugin import ReportWriterPlugin

        return ReportWriterPlugin
    if key == "RuntimeResourcePlugin":
        from .runtime_resource_plugin import RuntimeResourcePlugin

        return RuntimeResourcePlugin
    if key == "TrainerStateCheckpointPlugin":
        from .trainer_state_checkpoint_plugin import TrainerStateCheckpointPlugin

        return TrainerStateCheckpointPlugin
    raise AttributeError(f"module 'plugins' has no attribute '{key}'")
