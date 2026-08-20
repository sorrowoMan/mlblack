from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import build_solver
else:
    from .build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def _generate(model: Any) -> tuple[list[list[int]], dict[str, Any]]:
    import torch

    if model is None or not hasattr(model, "generate"):
        raise RuntimeError("language-model result does not expose generate(...)")
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    generated, cache = model.generate(input_ids, max_new_tokens=2, return_cache=True)
    return generated.detach().cpu().numpy().astype(int).tolist(), dict(cache)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tiny Transformer language-model Case")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "latest")
    args = parser.parse_args(argv)
    trainer = build_solver()
    if args.check:
        print_case_check(trainer)
        return 0
    result = trainer.fit(max_steps=max(1, int(args.steps)))
    tokens, cache = _generate(result.best_model)
    summary = {
        "case": "tiny_transformer_language_model",
        "best_score": result.report.get("best_score"),
        "best_metrics": result.report.get("best_metrics", {}),
        "generated_tokens": tokens,
        "kv_cache": {
            "schema": cache.get("schema"),
            "length": cache.get("length"),
            "num_layers": cache.get("num_layers"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
