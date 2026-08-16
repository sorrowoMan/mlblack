# START_HERE

## 1) Health Baseline
python -m mlblack project doctor --path . --strict

## 2) Run Comparison
python run_solver.py

## 3) Verify
python -m compileall -q .

## 4) Structure
- `build_solver.py`: main assembly entry -- imports all 7 temporal presets, runs comparison
- `pipeline/data_generator.py`: generates synthetic sine+noise time series data
- `run_solver.py`: simple CLI entrypoint delegates to build_solver.main()
