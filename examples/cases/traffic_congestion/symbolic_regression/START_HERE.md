# Traffic CI Symbolic Regression — START HERE

## Overview

Linear regression baseline on real traffic CI data. Uses `GradientDescentAdapter` + custom MSE problem + simple coefficient vector representation. This case establishes the baseline RMSE for the symbolic expression search approach.

## Component Composition

| Layer | Component | Source |
|---|---|---|
| Problem | CIDirectRegressionProblem | Custom (MSE + analytic gradient) |
| Representation | CIDirectRepresentation | Custom (flat coef vector) |
| Adapter | GradientDescentAdapter | Framework `adapter.gradient_descent` |
| Pipeline | ZScoreNormalizeComponent | Framework `pipeline` |

## Quick Start

```powershell
cd C:\Users\hp\Desktop\mlblack\examples\cases\traffic_congestion\symbolic_regression
python build_trainer.py
```

## Run Check Only (no fit)

```powershell
python build_trainer.py --check
```

## Run With Custom Steps

```powershell
python build_trainer.py --steps 50
python build_trainer.py --steps 500 --lr 0.005
```

## Verify

```powershell
python -m compileall -q .
```
