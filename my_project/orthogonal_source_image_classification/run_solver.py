from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from my_project.orthogonal_source_image_classification.build_solver import (  # noqa: E402
    ImageClassificationConfig,
    build_orthogonal_source_image_classification_components,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Orthogonal Source Layer -> image classification baseline scaffold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--suite-id", type=str, default="")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation-max-features", type=int, default=55)
    parser.add_argument("--representation-candidate-keep-top", type=int, default=120)
    parser.add_argument("--representation-max-pair-abs-corr", type=float, default=0.985)
    parser.add_argument("--max-sources", type=int, default=16)
    parser.add_argument("--candidate-keep-top", type=int, default=220)
    parser.add_argument("--max-pair-abs-corr", type=float, default=0.76)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs/orthogonal_source_image_classification")
    parser.add_argument("--check", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> ImageClassificationConfig:
    base = ImageClassificationConfig()
    keys = tuple(str(key) for key in args.datasets) if args.datasets else tuple(base.dataset_keys)
    return ImageClassificationConfig(
        dataset_keys=keys,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        representation_max_features=int(args.representation_max_features),
        representation_candidate_keep_top=int(args.representation_candidate_keep_top),
        representation_max_pair_abs_corr=float(args.representation_max_pair_abs_corr),
        max_sources=int(args.max_sources),
        candidate_keep_top=int(args.candidate_keep_top),
        max_pair_abs_corr=float(args.max_pair_abs_corr),
        max_rows=int(args.max_rows),
        output_dir=str(args.output_dir),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = _config_from_args(args)
    components = build_orthogonal_source_image_classification_components(cfg)
    if bool(args.check):
        print(
            "orthogonal_source_image_classification scaffold ok | "
            f"datasets={len(tuple(cfg.dataset_keys))} | "
            f"protocol={components['protocol']} | "
            f"component={components['orthogonal_component']}"
        )
        return
    result = components["runner"](cfg, suite_id=str(args.suite_id or None))
    print(f"[orth-image] suite_id={result['suite_id']}")
    print(f"[orth-image] output_dir={result['output_dir']}")
    print(f"[orth-image] table={result['artifacts']['classification_table_md']}")
    print(f"[orth-image] representation_table={result['artifacts']['representation_formula_table_csv']}")
    for row in result["rows"]:
        if row["feature_space"] in {
            "formula_pool",
            "image_representation",
            "orthogonal_sources",
            "image_representation_plus_orthogonal_sources",
        }:
            print(
                "[orth-image] "
                f"{row['scenario']} {row['feature_space']} {row['model']} "
                f"acc={float(row['test_accuracy']):.4f} "
                f"delta_vs_raw_pixels={row['accuracy_delta_vs_raw_pixels']} "
                f"delta_vs_image_repr={row['accuracy_delta_vs_image_representation']}"
            )


if __name__ == "__main__":
    main()
