"""Path bootstrap that ensures project/framework imports work."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_importable():
    """Add project root and framework parents to sys.path."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Ensure sibling framework packages are discoverable
    _nsga_parent = root.parent
    if (_nsga_parent / "nsgablack").is_dir() and str(_nsga_parent) not in sys.path:
        sys.path.insert(0, str(_nsga_parent))


def bootstrap():
    """Backward-compatible bootstrap entry used by run_project.py."""
    ensure_importable()
