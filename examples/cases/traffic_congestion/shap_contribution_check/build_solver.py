import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

DATA_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = DATA_DIR / "data" / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"


def _check_optional_deps():
    missing = []
    try:
        import xgboost
    except ImportError:
        missing.append("xgboost")
    try:
        import shap
    except ImportError:
        missing.append("shap")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        missing.append("scipy")
    return missing


def build_solver():
    """Canonical unified scaffold entry for this analysis case."""

    return main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200, help="XGBoost trees")
    parser.add_argument("--top-k", type=int, default=10, help="Top features to compare")
    args = parser.parse_args()

    missing = _check_optional_deps()
    if missing:
        print(f"Missing optional packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        print("Proceeding with available methods only.")

    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in ("date", "ci", "Unnamed: 0") and not c.startswith("test_fold_")]
    X = df[feature_cols].values.astype(float)
    y = df["ci"].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=" * 60)
    print("SHAP Contribution Consistency Check: Traffic CI")
    print(f"Samples: {len(y)}, Features: {len(feature_cols)}")
    print()

    lin = LinearRegression().fit(X_scaled, y)
    lin_coefs = lin.coef_
    lin_importance = np.abs(lin_coefs)
    lin_rank = np.argsort(-lin_importance)

    importance_methods = [("Linear", lin_importance)]

    has_xgb = "xgboost" not in missing
    has_shap = "shap" not in missing
    has_scipy = "scipy" not in missing

    if has_xgb:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=args.n_estimators, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_scaled, y)
        xgb_importance = xgb_model.feature_importances_
        importance_methods.append(("XGBoost", xgb_importance))

        if has_shap:
            import shap
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_scaled)
            shap_importance = np.abs(shap_values).mean(axis=0)
            importance_methods.append(("SHAP", shap_importance))

        perm_result = permutation_importance(
            xgb_model, X_scaled, y, n_repeats=5, random_state=42, scoring='neg_mean_squared_error'
        )
        perm_importance = perm_result.importances_mean
        importance_methods.append(("Permutation", perm_importance))
    else:
        print("[WARN] xgboost not available; using Linear only.")
        xgb_model = None

    n_methods = len(importance_methods)
    header = f"{'Feature':<30s}"
    rank_cols = [f"{m[0]:>8s} Rank" for m in importance_methods]
    header += "".join(rank_cols)
    header += f" {'Agreement':>10s}"
    print(header)
    print("-" * (40 + n_methods * 10))

    ranks = {}
    for label, imp in importance_methods:
        ranks[label] = np.argsort(-imp)

    agreements = []
    for i in range(min(args.top_k, len(feature_cols))):
        name = feature_cols[i][:28]
        line_parts = [f"{name:<30s}"]
        agree_count = 0
        for label, _ in importance_methods:
            r = int(np.where(ranks[label] == i)[0][0]) + 1
            line_parts.append(f"{r:>8d}")
            if r <= args.top_k:
                agree_count += 1
        marker_map = {4: "4/4", 3: "3/4", 2: "2/4", 1: "1/4"}
        marker = marker_map.get(agree_count, "1/4")
        line_parts.append(f"{marker:>10s}")
        agreements.append(agree_count)
        print("".join(line_parts))

    if n_methods > 1 and has_scipy:
        from scipy.stats import spearmanr
        print()
        print("Rank Correlations (Spearman):")
        method_labels = [m[0] for m in importance_methods]
        method_importances = [m[1] for m in importance_methods]
        for a in range(len(method_labels)):
            for b in range(a + 1, len(method_labels)):
                corr, p = spearmanr(method_importances[a], method_importances[b])
                print(f"  {method_labels[a]} vs {method_labels[b]}: {corr:.3f} (p={p:.4f})")
    else:
        print()
        print("Rank Correlations: scipy not available, skipping.")

    if agreements:
        avg_agreement = np.mean(agreements)
        print(f"\nAverage top-{args.top_k} agreement: {avg_agreement:.1f}/{n_methods} methods")

        if avg_agreement >= 3.0:
            print("Conclusion: Feature contributions are CONSISTENT across model paradigms.")
            print("  Linear attribution conclusions are robust.")
        elif avg_agreement >= 2.0:
            print("Conclusion: Feature contributions are MODERATELY consistent.")
            print("  Some features differ in importance by model type. Check individually.")
        else:
            print("Conclusion: Feature contributions DIFFER significantly across paradigms.")
            print("  Linear attribution may not generalize. Consider nonlinear models.")


if __name__ == "__main__":
    main()
