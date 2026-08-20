import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from types import SimpleNamespace
from typing import Mapping
import numpy as np

from mlblack.integrations import build_diagnostic_solver
from mlblack.project.scaffold import print_case_check

try:
    from .pipeline import build_pipeline
except ImportError:
    from pipeline import build_pipeline

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_PATH = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"

CAUSALITY_PAIRS = [
    ("wind", "ci", 7, "Wind -> CI"),
    ("ci", "wind", 7, "CI -> Wind"),
    ("aqi", "ci", 7, "AQI -> CI"),
    ("ci", "aqi", 7, "CI -> AQI"),
    ("is_bad_weather", "ci", 7, "Bad Weather -> CI"),
    ("ci", "is_bad_weather", 7, "CI -> Bad Weather"),
]

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    """Canonical unified scaffold entry for this analysis case."""

    payload = dict(config or {}) if isinstance(config, Mapping) else {}
    overrides = dict(component_overrides or {})
    def run_diagnostic(context):
        del context
        exit_code = _run_analysis(payload)
        return {"status": "ok", "exit_code": int(exit_code or 0)}

    runner = overrides.get("diagnostic_runner") or run_diagnostic
    return build_diagnostic_solver(
        runner,
        name="traffic_granger_causality_check",
        resource_context=resource_context,
    )


def _run_analysis(config=None):
    payload = dict(config or {})
    args = SimpleNamespace(
        maxlag=int(payload.get("maxlag", 7)),
        alpha=float(payload.get("alpha", 0.05)),
    )

    pipeline_data = build_pipeline(CSV_PATH)
    df = pipeline_data.frame

    print("=" * 60)
    print("Granger Causality Check: Traffic CI")
    print(f"Samples: {len(df)}, Max lag: {args.maxlag}, alpha={args.alpha}")
    print()

    factor_cols = list(pipeline_data.factor_columns)

    try:
        from statsmodels.tsa.stattools import grangercausalitytests, adfuller

        ci_cols = [c for c in df.columns if c.startswith("ci_lag") and c in df.columns]
        ci_features = ["ci"] + ci_cols[:3]

        print("Pairwise Granger Causality Tests:")
        print(f"{'Cause':<25s} {'Effect':<10s} {'Best Lag':>8s} {'F-stat':>10s} {'p-value':>10s} {'Significant?':>12s}")
        print("-" * 80)

        significant = []
        for factor in factor_cols:
            data = df[[factor, "ci"]].dropna()
            if len(data) < 20:
                continue

            x = data[factor].values
            y = data["ci"].values

            try:
                adf_p = adfuller(y, maxlag=min(args.maxlag, len(y)//4))[1]
            except Exception:
                adf_p = 0.0

            try:
                test_data = np.column_stack([y, x])
                result = grangercausalitytests(test_data, maxlag=min(args.maxlag, len(data)//10), verbose=False)
            except Exception:
                continue

            best_lag = 1
            best_p = 1.0
            best_f = 0.0
            for lag, res in result.items():
                ssr_ftest = res[0].get("ssr_ftest", (0.0, 1.0, 0.0, 0))
                p = ssr_ftest[1]
                if p < best_p:
                    best_p = p
                    best_lag = lag
                    best_f = ssr_ftest[0]

            is_sig = "YES" if best_p < args.alpha else "no"
            short_factor = factor[:22] if len(factor) > 22 else factor
            print(f"{short_factor + ' -> CI':<25s} {'ci':<10s} {best_lag:>8d} {best_f:>10.2f} {best_p:>10.4f} {is_sig:>12s}")

            if best_p < args.alpha:
                significant.append((factor, best_lag, best_p, best_f))

        print()
        print("Reverse Direction (CI -> factors):")
        print(f"{'Effect':<25s} {'Cause':<10s} {'Best Lag':>8s} {'F-stat':>10s} {'p-value':>10s} {'Significant?':>12s}")
        print("-" * 80)

        for factor in factor_cols:
            data = df[[factor, "ci"]].dropna()
            if len(data) < 20:
                continue

            x = data["ci"].values
            y = data[factor].values

            try:
                test_data = np.column_stack([y, x])
                result = grangercausalitytests(test_data, maxlag=min(args.maxlag, len(data)//10), verbose=False)
            except Exception:
                continue

            best_lag = 1
            best_p = 1.0
            best_f = 0.0
            for lag, res in result.items():
                ssr_ftest = res[0].get("ssr_ftest", (0.0, 1.0, 0.0, 0))
                p = ssr_ftest[1]
                if p < best_p:
                    best_p = p
                    best_lag = lag
                    best_f = ssr_ftest[0]

            is_sig = "YES" if best_p < args.alpha else "no"
            short_factor = factor[:22] if len(factor) > 22 else factor
            print(f"CI -> {short_factor:<22s} {'ci':<10s} {best_lag:>8d} {best_f:>10.2f} {best_p:>10.4f} {is_sig:>12s}")

        print()
        print("Multi-variate VAR Granger Test:")
        try:
            from statsmodels.tsa.api import VAR
            var_features = ["ci"] + [f for f, _, _, _ in significant[:4]]
            if len(var_features) < 2:
                var_features = ["ci", "wind", "aqi"]
            var_data = df[var_features].dropna()

            var_model = VAR(var_data.values)
            var_result = var_model.fit(maxlags=min(args.maxlag, len(var_data)//20))

            print(f"  Variables: {', '.join(var_features)}")
            print(f"  Selected lag order: {var_result.k_ar}")
            print(f"  AIC: {var_result.aic:.1f}")
        except Exception as e:
            print(f"  VAR skipped: {e}")

        print()
        if significant:
            print("Significant Granger-causal relationships found:")
            for factor, lag, p, f in sorted(significant, key=lambda x: x[2]):
                print(f"  {factor} -> CI  (lag={lag}, F={f:.1f}, p={p:.4f})")
        else:
            print(f"No significant Granger-causal relationships at alpha={args.alpha}")

    except ImportError:
        print("statsmodels not installed. Install: pip install statsmodels")
        print()
        print("Falling back to cross-correlation analysis...")

        from scipy import signal
        ci = df["ci"].values

        print("Cross-correlation with CI (max correlation lag):")
        for factor in factor_cols[:8]:
            if factor in df.columns:
                x = df[factor].dropna().values
                y_clean = ci[~np.isnan(x)]
                x_clean = x[~np.isnan(x)]
                if len(x_clean) > 20:
                    corr = np.correlate(x_clean - x_clean.mean(), y_clean - y_clean.mean(), mode='full')
                    corr /= (len(x_clean) * x_clean.std() * y_clean.std())
                    lags = signal.correlation_lags(len(x_clean), len(y_clean))
                    max_idx = np.argmax(np.abs(corr))
                    print(f"  {factor[:30]:<30s}: lag={lags[max_idx]:>4d}, corr={corr[max_idx]:+.3f}")

    return 0
