import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"

FACTOR_GROUPS = {
    "Weather": ["weather_dummy", "wind", "is_bad_weather"],
    "AQI": ["aqi", "is_aqi_high"],
    "Holiday": ["is_holiday_near", "is_holiday_mid", "is_nonwork_weekend", "is_holiday_day_or_window"],
    "Life": ["life_impact"],
    "CI_Lags": ["ci_lag1", "ci_lag2", "ci_lag3", "ci_lag7", "ci_lag8", "ci_lag14", "ci_lag21", "ci_lag28"],
    "CI_Rolling": ["ci_roll3_prev_mean", "ci_roll7_prev_mean", "ci_roll14_prev_mean", "ci_roll7_prev_std"],
    "Time_Cyclic": ["dow_sin", "dow_cos", "doy_sin", "doy_cos"],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-order", type=int, default=2, help="AR order")
    parser.add_argument("--ma-order", type=int, default=1, help="MA order")
    parser.add_argument("--diff", type=int, default=0, help="Differencing order")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    exclude = ("date", "ci", "Unnamed: 0", "test_fold_1", "test_fold_2", "test_fold_3",
               "test_fold_4", "test_fold_5", "test_fold_6", "test_fold_7", "test_fold_8",
               "test_fold_9", "test_fold_10", "dow", "month")
    all_features = [c for c in df.columns if c not in exclude and not c.startswith("test_fold_")]

    available_groups = {}
    for gname, features in FACTOR_GROUPS.items():
        present = [f for f in features if f in all_features]
        if present:
            available_groups[gname] = present

    X_all = df[[f for g in available_groups.values() for f in g]].values.astype(float)
    y = df["ci"].values.astype(float)

    mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y)
    X_all = X_all[mask]
    y = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    print("=" * 60)
    print("ARIMAX Factor Attribution: Traffic CI")
    print(f"Samples: {len(y)}, External features: {X_all.shape[1]}")
    print(f"ARIMA({args.ar_order},{args.diff},{args.ma_order})")
    print()

    try:
        import statsmodels.api as sm
        from statsmodels.tsa.arima.model import ARIMA

        full_exog = X_scaled
        full_model = ARIMA(y, order=(args.ar_order, args.diff, args.ma_order), exog=full_exog)
        full_result = full_model.fit()
        full_aic = full_result.aic
        full_bic = full_result.bic

        n_exog = full_exog.shape[1]
        exog_coefs = full_result.params[-n_exog:]

        print(f"Full ARIMAX: AIC={full_aic:.1f}, BIC={full_bic:.1f}")
        print()

        print("Factor Group Contributions (standardized):")
        print(f"{'Group':<20s} {'|Coef| Sum':>12s} {'Contribution %':>14s}")
        print("-" * 48)

        total_abs = np.sum(np.abs(exog_coefs))
        col_offset = 0
        group_contributions = {}
        for gname, features in available_groups.items():
            nf = len(features)
            group_coefs = exog_coefs[col_offset:col_offset + nf]
            abs_sum = np.sum(np.abs(group_coefs))
            pct = abs_sum / total_abs * 100 if total_abs > 0 else 0
            group_contributions[gname] = (abs_sum, pct, features, group_coefs)
            print(f"{gname:<20s} {abs_sum:>12.4f} {pct:>13.1f}%")
            col_offset += nf

        print()
        print("Factor Removal Impact (drop-one-group):")
        print(f"{'Dropped Group':<20s} {'AIC':>10s} {'Delta AIC':>10s} {'Impact':>10s}")
        print("-" * 52)
        print(f"{'(none - full)':<20s} {full_aic:>10.1f} {'-':>10s} {'-':>10s}")

        impacts = []
        col_start = 0
        for gname, features in available_groups.items():
            nf = len(features)
            mask_cols = np.ones(n_exog, dtype=bool)
            mask_cols[col_start:col_start + nf] = False
            reduced_exog = full_exog[:, mask_cols]
            reduced_model = ARIMA(y, order=(args.ar_order, args.diff, args.ma_order), exog=reduced_exog)
            try:
                reduced_result = reduced_model.fit()
                reduced_aic = reduced_result.aic
                delta = reduced_aic - full_aic
                impact = "HIGH" if delta > 4 else ("MED" if delta > 2 else "LOW")
                impacts.append((gname, delta, impact))
                print(f"{gname:<20s} {reduced_aic:>10.1f} {delta:>+10.1f} {impact:>10s}")
            except Exception:
                print(f"{gname:<20s} {'FAIL':>10s} {'-':>10s} {'-':>10s}")
            col_start += nf

        print()
        print("Interpretation:")
        print("  - High delta AIC (>4): group is critical for prediction")
        print("  - Medium delta (2-4): group has moderate contribution")
        print("  - Low delta (<2): group can be dropped with minimal impact")

        if impacts:
            critical = [g for g, d, i in impacts if i == "HIGH"]
            if critical:
                print(f"  Critical factor groups: {', '.join(critical)}")

    except ImportError:
        print("statsmodels not installed. Install: pip install statsmodels")
        print("Falling back to LinearRegression factor contribution analysis...")
        print()

        from sklearn.linear_model import LinearRegression

        lin = LinearRegression()
        lin.fit(X_scaled, y)

        full_r2 = lin.score(X_scaled, y)
        coefs = lin.coef_
        total_abs = np.sum(np.abs(coefs))

        print(f"Linear R^2: {full_r2:.4f}")
        print()
        print(f"{'Group':<20s} {'|Coef| Sum':>12s} {'Contribution %':>14s}")
        print("-" * 48)

        col_offset = 0
        for gname, features in available_groups.items():
            nf = len(features)
            group_coefs = coefs[col_offset:col_offset + nf]
            abs_sum = np.sum(np.abs(group_coefs))
            pct = abs_sum / total_abs * 100 if total_abs > 0 else 0
            print(f"{gname:<20s} {abs_sum:>12.4f} {pct:>13.1f}%")
            col_offset += nf

if __name__ == "__main__":
    main()
