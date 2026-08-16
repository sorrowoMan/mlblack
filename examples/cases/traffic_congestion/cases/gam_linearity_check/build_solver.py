import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import time
from types import SimpleNamespace
from typing import Mapping
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV

from mlblack.core import build_diagnostic_trainer
from mlblack.project.scaffold import print_case_check

try:
    from .pipeline import build_pipeline
except ImportError:
    from pipeline import build_pipeline

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_PATH = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    """Canonical unified scaffold entry for this analysis case."""

    payload = dict(config or {}) if isinstance(config, Mapping) else {}
    overrides = dict(component_overrides or {})
    def run_diagnostic(context):
        del context
        exit_code = _run_analysis(payload)
        return {"status": "ok", "exit_code": int(exit_code or 0)}

    runner = overrides.get("diagnostic_runner") or run_diagnostic
    return build_diagnostic_trainer(
        runner,
        name="traffic_gam_linearity_check",
        resource_context=resource_context,
    )


def _run_analysis(config=None):
    payload = dict(config or {})
    args = SimpleNamespace(
        n_knots=int(payload.get("n_knots", 6)),
        degree=int(payload.get("degree", 3)),
        top_k=int(payload.get("top_k", 8)),
    )

    data = build_pipeline(CSV_PATH)
    feature_cols = list(data.feature_names)
    X_scaled = np.asarray(data.X_train, dtype=float)
    y = np.asarray(data.y_train, dtype=float)
    
    print("=" * 60)
    print("GAM Linearity Check: Traffic CI")
    print(f"Samples: {len(y)}, Features: {len(feature_cols)}")
    print(f"B-spline: {args.n_knots} knots, degree={args.degree}")
    print()
    
    # --- Linear baseline ---
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression()
    lin.fit(X_scaled, y)
    lin_pred = lin.predict(X_scaled)
    lin_rmse = np.sqrt(np.mean((y - lin_pred)**2))
    lin_coefs = lin.coef_
    
    # --- GAM: B-spline for ALL continuous features + linear for binary ---
    # Build GAM by spline-transforming each feature, then Ridge regression
    spline_parts = []
    feature_map = {}  # feature_idx -> (start_spline_col, n_spline_cols, type, fitted_spline)
    col = 0
    for i, name in enumerate(feature_cols):
        vals = X_scaled[:, i].reshape(-1, 1)
        unique_vals = len(np.unique(np.round(vals, 3)))
        if unique_vals <= 3:
            # Binary/categorical: keep linear
            spline_parts.append(vals)
            feature_map[i] = (col, 1, "binary", None)
            col += 1
        else:
            n_knots_actual = min(args.n_knots, max(3, unique_vals // 5))
            spline = SplineTransformer(n_knots=n_knots_actual, degree=args.degree, include_bias=False)
            transformed = spline.fit_transform(vals)
            spline_parts.append(transformed)
            feature_map[i] = (col, transformed.shape[1], "spline", spline)
            col += transformed.shape[1]
    
    X_gam = np.hstack(spline_parts)
    gam_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    gam_model.fit(X_gam, y)
    gam_pred = gam_model.predict(X_gam)
    gam_rmse = np.sqrt(np.mean((y - gam_pred)**2))
    
    # --- Partial dependence comparison ---
    print(f"{'Feature':<30s} {'Type':<8s} {'Linear Coef':>12s} {'GAM Range':>12s} {'Nonlinear?':>10s}")
    print("-" * 75)
    
    nonlinear_features = []
    for i, name in enumerate(feature_cols):
        start, n_cols, ftype, spline_fitted = feature_map[i]
        lin_c = lin_coefs[i]
        
        if ftype == "binary":
            print(f"{name:<30s} {'binary':<8s} {lin_c:>+12.4f} {'N/A':>12s} {'N/A':>10s}")
        else:
            # GAM partial dependence: evaluate at 100 quantile points using fitted spline
            x_grid = np.linspace(X_scaled[:, i].min(), X_scaled[:, i].max(), 100).reshape(-1, 1)
            gam_vals = np.dot(gam_model.coef_[start:start+n_cols], spline_fitted.transform(x_grid).T)
            x_flat = x_grid.ravel()
            lin_proj = x_flat * lin_c
            
            gam_range = f"{gam_vals.min():+.4f} ~ {gam_vals.max():+.4f}"
            lin_range = f"{lin_proj.min():.4f} ~ {lin_proj.max():.4f}"
            
            # Nonlinearity check: GAM range width vs linear range width
            gam_span = gam_vals.max() - gam_vals.min()
            lin_span = abs(x_flat.max() - x_flat.min()) * abs(lin_c)
            is_nonlinear = "YES" if (gam_span > 1.5 * lin_span or np.corrcoef(lin_proj, gam_vals)[0,1] < 0.8) else "no"
            
            print(f"{name:<30s} {'spline':<8s} {lin_c:>+12.4f} {gam_range:>12s} {is_nonlinear:>10s}")
            if is_nonlinear == "YES":
                nonlinear_features.append((name, lin_c, gam_span, lin_span))
    
    print()
    print(f"Linear RMSE: {lin_rmse:.4f}")
    print(f"GAM+B-spline RMSE: {gam_rmse:.4f}")
    print(f"RMSE improvement: {(1 - gam_rmse/lin_rmse)*100:.1f}%")
    print()
    
    if nonlinear_features:
        print(f"Nonlinear features detected ({len(nonlinear_features)}):")
        for name, lc, gs, ls in sorted(nonlinear_features, key=lambda x: -x[2]):
            print(f"  {name}: GAM span={gs:.2f} vs Linear span={ls:.2f} ({gs/ls:.1f}x)")
    else:
        print("No strong nonlinearity detected. Linear assumptions broadly hold.")
    
    print()
    print("Conclusion: GAM diagnoses whether linear model assumptions are justified.")
    print("If GAM RMSE >> Linear RMSE, nonlinear patterns exist that linear model misses.")
    return 0
