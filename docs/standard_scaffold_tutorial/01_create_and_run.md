# 01. Create And Run

## 1) Create project

```powershell
python -m nsgablack project new my_ml_project
cd my_ml_project
```

## 2) Add cases by semantic role

```powershell
python -m nsgablack project add-case my_trainer --type trainer --framework mlblack
python -m nsgablack project add-case outer_search --type solver --framework nsgablack
```

## 3) Add components

```powershell
# pipeline entry or operator
python -m nsgablack project add-component --case my_trainer --kind pipeline --slot main --name main
python -m nsgablack project add-component --case my_trainer --kind pipeline --slot codec --name linear_codec
python -m nsgablack project add-component --case my_trainer --kind pipeline --slot head --name point_head

# adapter / bias / plugin
python -m nsgablack project add-component --case my_trainer --kind adapter --name trainer_adapter
python -m nsgablack project add-component --case my_trainer --kind bias --name objective_bias
python -m nsgablack project add-component --case my_trainer --kind plugin --name resource_audit_plugin
```

## 4) Validate and run

```powershell
python run_project.py --check --build-check
python -m mlblack project doctor --path . --strict
python run_project.py
```
