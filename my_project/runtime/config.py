from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLBLACK scaffold runtime workflow")
    parser.add_argument("--run-id", type=str, default="", help="Optional run id")
    return parser
