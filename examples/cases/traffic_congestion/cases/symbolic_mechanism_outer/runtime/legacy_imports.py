from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "legacy_nowcasting"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _cur in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    if (_cur / "mlblack.py").is_file() and (_cur / "pyproject.toml").is_file():
        _parent = str(_cur.parent)
        if _parent not in sys.path:
            sys.path.append(_parent)
        break

for _candidate in (
    Path.home() / "Desktop" / "nsgablack",
    PROJECT_ROOT.parents[2] / "nsgablack" if len(PROJECT_ROOT.parents) > 2 else None,
):
    if _candidate is not None and (_candidate / "__init__.py").is_file() and (_candidate / "core").is_dir():
        _parent = str(_candidate.parent)
        if _parent not in sys.path:
            sys.path.append(_parent)
        break

from core.common.contracts import ProcessedDataset
from core.models.symbolic_torch_model import SymbolicTorchRegressor
from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.symbolic.gradient_parser import GradientParser
from core.symbolic.symbolic_dsl import evaluate_genome_numpy
from core.symbolic.symbolic_structure_search import evaluate_genome_with_ridge
from core.trainers.xgboost_trainer import XGBoostSurrogateTrainer, XGBoostTrainerConfig
from legacy_nowcasting.examples.path_defaults import default_work_ci_csv
from legacy_nowcasting.examples.work_ci_reader import WorkCiIntervalReader
from nsgablack.adapters import (
    MOEADAdapter,
    MOEADConfig,
    NSGA2Adapter,
    NSGA2Config,
    SerialPhaseSpec,
    SerialStrategyConfig,
    StrategyChainAdapter,
    VNSAdapter,
    VNSConfig,
)
from nsgablack.core.base import BlackBoxProblem
from nsgablack.core.composable_solver import ComposableSolver
from nsgablack.representation import RepresentationPipeline
from nsgablack.representation.continuous import ClipRepair, ContextGaussianMutation, UniformInitializer

