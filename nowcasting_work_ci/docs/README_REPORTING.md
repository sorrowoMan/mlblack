# Result Aggregation & Visualization

This package now includes a reporting utility:

- [tools/aggregate_and_plot_results.py](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/tools/aggregate_and_plot_results.py)

## What it does

1. Scans `_scenario_runs/nowcasting_work_ci/**/summary.json`
2. Optionally scans top-level sweep/aggregate JSON files in that scenario runs root
3. Builds:
   - flat run table (`runs_flat.csv/.md`)
   - grouped experiment table (`experiment_aggregate.csv/.md`)
4. Draws:
   - `picp_vs_pinaw.png` (core interval-quality tradeoff)
   - `rmse_vs_pinaw.png` (point-vs-interval compromise)
5. Writes `dashboard.json` with key pointers

## Usage

```powershell
python C:\Users\hp\Desktop\mlblack\nowcasting_work_ci\tools\aggregate_and_plot_results.py
```

Optional:

```powershell
python C:\Users\hp\Desktop\mlblack\nowcasting_work_ci\tools\aggregate_and_plot_results.py `
  --out-root C:\Users\hp\Desktop\mlblack\_scenario_runs\nowcasting_work_ci `
  --report-dir C:\Users\hp\Desktop\mlblack\_scenario_runs\nowcasting_work_ci\reports\my_snapshot `
  --include-sweep-json 1
```

## Output location

Default output:

- `C:\Users\hp\Desktop\mlblack\_scenario_runs\nowcasting_work_ci\reports\<timestamp>\`
