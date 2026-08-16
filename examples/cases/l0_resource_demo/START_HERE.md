# START_HERE

## 1. Add your first case

```powershell
python -m nsgablack project add-case my_first_solver --type solver
```

## 2. Edit the case

Open `cases/my_first_solver/build_solver.py` and implement `build_solver()`.
Keep `pipeline/main.py` as the single pipeline entry, and add fine-grained
operators under `pipeline/operators/*`.

## 3. Validate from the Project root

```powershell
python run_project.py --check --build-check
```

## 4. Run

```powershell
python run_project.py
```

## 5. Debug one Case only

```powershell
cd cases/my_first_solver
python run_solver.py --check
cd ../..
```

`run_solver.py` is only a case-local debug entry. Formal runs start at
`run_project.py` so Project L0 can grant the effective `ResourceContext`.

## 6. Health check

```powershell
python -m nsgablack project doctor --path . --build --strict
```

## 7. Add pipeline operators quickly

```powershell
python -m nsgablack project add-component --case my_first_solver --kind pipeline --slot mutate --name gaussian_mutate
python -m nsgablack project add-component --case my_first_solver --kind pipeline --slot repair --name clip_repair
```
