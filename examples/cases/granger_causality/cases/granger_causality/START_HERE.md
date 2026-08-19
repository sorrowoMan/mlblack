# Granger Causality — START HERE

## Overview

This case demonstrates Granger causality testing as sparse VAR(1) coefficient optimization — a pure mlblack gradient-descent learning problem.

## Component Composition

| Layer | Component | Source |
|---|---|---|
| Problem | GrangerCausalityProblem | Custom (VAR MSE + L1 sparsity + analytic gradient) |
| Representation | GrangerRepresentation | Custom (flat vector ↔ A matrix) |
| Adapter | GradientOptimizerAdapter | Stable method `gradient.sgd` |

## Quick Start

```powershell
python examples\cases\granger_causality\run_project.py --check --build-check
python examples\cases\granger_causality\cases\granger_causality\run_solver.py
```

## Run Check Only (no fit)

```powershell
python examples\cases\granger_causality\run_project.py --check --build-check
```

## Verify

```powershell
python -m compileall -q .
```
