# Granger Causality — START HERE

## Overview

This case demonstrates Granger causality testing as sparse VAR(1) coefficient optimization — a pure mlblack gradient-descent learning problem.

## Component Composition

| Layer | Component | Source |
|---|---|---|
| Problem | GrangerCausalityProblem | Custom (VAR MSE + L1 sparsity + analytic gradient) |
| Representation | GrangerRepresentation | Custom (flat vector ↔ A matrix) |
| Adapter | GradientDescentAdapter | Framework `adapter.gradient_descent` |

## Quick Start

```powershell
python build_trainer.py
```

## Run Check Only (no fit)

```powershell
python build_trainer.py --check
```

## Verify

```powershell
python -m compileall -q .
```
