# Traffic CI Symbolic Regression — START HERE

## Overview

Linear regression baseline on real traffic CI data. Uses stable `gradient.sgd` + custom MSE problem + simple coefficient vector representation. This case establishes the baseline RMSE for the symbolic expression search approach.

## Component Composition

| Layer | Component | Source |
|---|---|---|
| Problem | CIDirectRegressionProblem | Custom (MSE + analytic gradient) |
| Representation | CIDirectRepresentation | Custom (flat coef vector) |
| Adapter | GradientOptimizerAdapter | Stable method `gradient.sgd` |
| Pipeline | ZScoreNormalizeComponent | Framework `pipeline` |

## Quick Start

```powershell
cd C:\Users\hp\Desktop\mlblack
python examples\cases\traffic_congestion\run_project.py --check
```

## Run Check Only (no fit)

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
```

## Run With Custom Steps

```powershell
python examples\cases\traffic_congestion\cases\symbolic_regression\run_solver.py --steps 50
python examples\cases\traffic_congestion\cases\symbolic_regression\run_solver.py --steps 500 --lr 0.005
```

## Verify

```powershell
python -m compileall -q .
```
