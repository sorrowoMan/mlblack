"""Scaffold package delegates to the unified nsgablack scaffold.

All project/case scaffolding is now handled by nsgablack's scaffold system.
mlblack cases use the identical unified template as nsgablack solver cases.
The only difference is catalog registration kind (solver vs trainer).
"""

from nsgablack.project.scaffold import add_case, create_project  # noqa: F401

__all__ = ["add_case", "create_project"]
