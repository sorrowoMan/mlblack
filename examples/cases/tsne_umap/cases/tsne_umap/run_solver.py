"""Canonical CLI for the t-SNE Case."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from blackbase.project import load_case_builder
from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="t-SNE with sklearn digits")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=200.0)
    parser.add_argument("--exaggeration", type=float, default=4.0)
    parser.add_argument("--exaggeration-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    case_root = Path(__file__).resolve().parent
    builder = load_case_builder(case_root.parent.parent, case_root.name, case_kind="trainer")
    trainer = builder(
        perplexity=args.perplexity,
        learning_rate=args.lr,
        exaggeration=args.exaggeration,
        exaggeration_steps=args.exaggeration_steps,
    )
    if args.check:
        print_case_check(trainer)
        return 0
    started = time.perf_counter()
    result = trainer.fit(max_steps=args.steps)
    elapsed = time.perf_counter() - started
    final_kl = result.best_feedback.loss if result.best_feedback else float("nan")
    if final_kl is None and result.best_feedback:
        final_kl = float(result.best_feedback.metrics.get("kl_divergence", float("nan")))
    print(f"elapsed: {elapsed:.2f}s")
    print(f"final KL divergence: {final_kl:.6f}")
    print(f"steps completed: {len(result.history)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
