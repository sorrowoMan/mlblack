from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

def _bootstrap() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mlblack").is_dir():
            sys.path.insert(0, str(parent))
            return


_bootstrap()

from mlblack.core import ArtifactBuilder, save_artifact_html  # noqa: E402
from mlblack.presets import (  # noqa: E402
    build_tiny_transformer_classification_trainer,
    build_tiny_transformer_dpo_preference_trainer,
    build_tiny_transformer_lm_trainer,
)

if __package__ in {None, ""}:
    from pipeline.main import build_classification_data, build_lm_data, build_preference_data  # noqa: E402
else:
    from .pipeline.main import build_classification_data, build_lm_data, build_preference_data  # noqa: E402


def run_case(*, steps: int, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    classification = build_tiny_transformer_classification_trainer(
        build_classification_data(),
        vocab_size=16,
        max_length=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        learning_rate=1e-2,
        random_seed=5,
        run_name="tiny_transformer_classification_case",
    )
    classification_result = classification.fit(max_steps=int(steps))
    classification_bundle = ArtifactBuilder().build(classification, classification_result)
    html_path = save_artifact_html(
        classification_bundle,
        output_dir / "classification_artifact.html",
        title="tiny transformer classification artifact",
    )

    lm = build_tiny_transformer_lm_trainer(
        build_lm_data(),
        vocab_size=12,
        max_length=5,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_kind="swiglu",
        norm="rms_norm",
        position_encoding="rope",
        learning_rate=1e-2,
        random_seed=9,
        run_name="tiny_transformer_lm_case",
    )
    lm_result = lm.fit(max_steps=int(steps))
    generated_tokens, kv_cache = _generate_from_lm(lm_result.best_model)

    preference = build_tiny_transformer_dpo_preference_trainer(
        build_preference_data(),
        vocab_size=10,
        max_length=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        qlora={"rank": 2, "bits": 4, "targets": ("attention.q", "attention.v")},
        learning_rate=1e-2,
        random_seed=21,
        run_name="tiny_transformer_dpo_case",
    )
    preference_result = preference.fit(max_steps=1)

    summary = {
        "case": "tiny_transformer_smoke",
        "steps": int(steps),
        "classification": {
            "best_score": classification_result.report.get("best_score"),
            "best_metrics": classification_result.report.get("best_metrics", {}),
            "artifact_html": str(html_path),
        },
        "language_model": {
            "best_score": lm_result.report.get("best_score"),
            "best_metrics": lm_result.report.get("best_metrics", {}),
            "generated_tokens": generated_tokens,
            "kv_cache": {
                "schema": kv_cache.get("schema"),
                "length": kv_cache.get("length"),
                "num_layers": kv_cache.get("num_layers"),
            },
        },
        "preference_dpo": {
            "best_score": preference_result.report.get("best_score"),
            "best_metrics": preference_result.report.get("best_metrics", {}),
            "adapter": preference_result.report.get("adapter", {}).get("name"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _generate_from_lm(model: Any) -> tuple[list[list[int]], dict[str, Any]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tiny transformer generation example requires torch") from exc
    if model is None or not hasattr(model, "generate"):
        raise RuntimeError("LM result did not expose a tiny transformer model with generate(...)")
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    generated, cache = model.generate(input_ids, max_new_tokens=2, return_cache=True)
    return generated.detach().cpu().numpy().astype(int).tolist(), dict(cache)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny mlblack Transformer smoke case.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "latest",
    )
    args = parser.parse_args()
    summary = run_case(steps=max(1, int(args.steps)), output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
