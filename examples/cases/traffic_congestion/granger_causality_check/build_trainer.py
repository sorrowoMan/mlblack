import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"

CAUSALITY_PAIRS = [
    ("wind", "ci", 7, "Wind -> CI"),
    ("ci", "wind", 7, "CI -> Wind"),
    ("aqi", "ci", 7, "AQI -> CI"),
    ("ci", "aqi", 7, "CI -> AQI"),
    ("is_bad_weather", "ci", 7, "Bad Weather -> CI"),
    ("ci", "is_bad_weather", 7, "CI -> Bad Weather"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxlag", type=int, default=7, help="Maximum lag for Granger test")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)

    print("=" * 60)
    print("Granger Causality Check: Traffic CI")
    print(f"Samples: {len(df)}, Max lag: {args.maxlag}, alpha={args.alpha}")
    print()

    exclude = ("date", "ci", "Unnamed: 0", "dow", "month")
    factor_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("test_fold_"):
            continue
        if col in ("weather_dummy", "wind", "aqi", "life_impact", "is_bad_weather",
                   "is_aqi_high", "is_holiday_near", "is_holiday_mid",
                   "is_nonwork_weekend", "is_holiday_day_or_window"):
            factor_cols.append(col)

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

if __name__ == "__main__":
    main()
