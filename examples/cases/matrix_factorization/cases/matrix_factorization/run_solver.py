# -*- coding: utf-8 -*-
# CLI entrypoint for running the matrix factorization solver/trainer case.

from __future__ import annotations

# CLI contract: --check builds the real Trainer without fitting.

if __name__ == "__main__":
    from build_solver import main
    main()
