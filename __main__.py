#!/usr/bin/env python
"""Module entrypoint for ``python -m mlblack``."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
