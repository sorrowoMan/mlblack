# START_HERE

## 1) Health Baseline
python -m mlblack project doctor --path . --strict

## 2) Define the Core Layers
- problem/: evaluate model -> produce Feedback
- pipeline/: prepare data for the problem
- pipeline/representation/: encode/decode unknown state + head output
- adapter/: propose/update optimization strategy

## 3) Wire the Assembly
- build_solver.py is the canonical assembly entry; build_trainer.py is an alias

## 4) Run
python run_solver.py

## 5) Verify
python -m compileall -q .
