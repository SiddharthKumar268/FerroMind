"""
evaluate.py — FerroMind Model Evaluation
SAIL Durgapur Steel Plant | Steel Melting Shop

Usage:
    python src/evaluate.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
TEST_CSV    = os.path.join("data", "processed", "sms_test.csv")
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# EXACT column names from your CSVs
# ─────────────────────────────────────────────
TARGET_COLS       = ["Mn", "Si", "C", "S", "P"]
LOG_TRANSFORM_COL = "C"
LOG_TRANSFORM_IDX = TARGET_COLS.index(LOG_TRANSFORM_COL)


# ─────────────────────────────────────────────
# LOAD TEST DATA
# ─────────────────────────────────────────────
def load_test_data(feature_names):
    print(f"Loading {TEST_CSV} ...")
    df = pd.read_csv(TEST_CSV)
    available = [c for c in feature_names if c in df.columns]
    X = df[available].fillna(df[available].median(numeric_only=True))
    y = df[TARGET_COLS].copy()
    print(f"  Test rows: {len(X):,}")
    return X, y, df


# ─────────────────────────────────────────────
# PREDICT — inverse log transform C
# ─────────────────────────────────────────────
def predict(model, X_scaled):
    preds = model.predict(X_scaled)
    preds[:, LOG_TRANSFORM_IDX] = np.expm1(preds[:, LOG_TRANSFORM_IDX])
    return preds


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    rows = []
    for i, col in enumerate(TARGET_COLS):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        rows.append({"Element": col, "RMSE": rmse, "MAE": mae, "R2": r2})
    return pd.DataFrame(rows).set_index("Element")


def compute_per_grade_rmse(y_true, y_pred, grade_series):
    df = pd.DataFrame(y_true, columns=[f"{c}_true" for c in TARGET_COLS])
    for i, c in enumerate(TARGET_COLS):
        df[f"{c}_pred"] = y_pred[:, i]
    df["BOF Grade Code"] = grade_series.values

    rows = []
    for grade, grp in df.groupby("BOF Grade Code"):
        row = {"Grade": grade, "N_heats": len(grp)}
        for col in TARGET_COLS:
            rmse = np.sqrt(mean_squared_error(grp[f"{col}_true"], grp[f"{col}_pred"]))
            row[f"RMSE_{col}"] = round(rmse, 5)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("N_heats", ascending=False)


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name):
    importances = []
    for est in model.estimators_:
        if hasattr(est, "feature_importances_"):
            importances.append(est.feature_importances_)
    if not importances:
        print(f"  [WARN] No feature_importances_ for {model_name}")
        return

    avg_imp = np.mean(importances, axis=0)
    idx = np.argsort(avg_imp)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx[:20]][::-1],
        avg_imp[idx[:20]][::-1],
        color="#2563eb",
    )
    ax.set_xlabel("Mean Importance (avg across 5 targets)")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(REPORTS_DIR, f"feature_importance_{model_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────
def print_metrics_table(metrics_df, model_name):
    print(f"\n  ── {model_name} ──")
    print(f"  {'Element':<10} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    print(f"  {'-'*40}")
    for el, row in metrics_df.iterrows():
        print(f"  {el:<10} {row['RMSE']:>10.5f} {row['MAE']:>10.5f} {row['R2']:>8.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FerroMind — Model Evaluation")
    print("=" * 60)

    scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    X, y, df_test = load_test_data(feature_names)
    X_scaled = scaler.transform(X)
    y_true   = y.values

    model_files = {
        "linear_regression"    : "linear_regression.pkl",
        "random_forest"        : "random_forest.pkl",
        "xgboost_composition"  : "xgboost_composition.pkl",
        "lightgbm_composition" : "lightgbm_composition.pkl",
    }

    all_metrics = {}
    preds_store = {}

    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"\n  [SKIP] {path} not found")
            continue
        print(f"\nEvaluating {name} ...")
        model = joblib.load(path)
        preds = predict(model, X_scaled)
        preds_store[name] = preds
        metrics = compute_metrics(y_true, preds)
        all_metrics[name] = metrics
        print_metrics_table(metrics, name)

    # Ensemble XGB + LGBM
    if "xgboost_composition" in preds_store and "lightgbm_composition" in preds_store:
        print("\nComputing XGBoost + LightGBM ensemble...")
        ens_preds = (preds_store["xgboost_composition"] + preds_store["lightgbm_composition"]) / 2.0
        ens_metrics = compute_metrics(y_true, ens_preds)
        all_metrics["ensemble_xgb_lgbm"] = ens_metrics
        preds_store["ensemble_xgb_lgbm"] = ens_preds
        print_metrics_table(ens_metrics, "Ensemble (XGB + LGBM)")

    # Save summary CSV
    rows = []
    for mname, mdf in all_metrics.items():
        for el, row in mdf.iterrows():
            rows.append({"Model": mname, "Element": el,
                         "RMSE": row["RMSE"], "MAE": row["MAE"], "R2": row["R2"]})
    pd.DataFrame(rows).to_csv(os.path.join(REPORTS_DIR, "metrics_summary.csv"), index=False)
    print(f"\n  Metrics saved → reports/metrics_summary.csv")

    # Per-grade RMSE
    if "xgboost_composition" in preds_store and "BOF Grade Code" in df_test.columns:
        print("\nComputing per-grade RMSE for xgboost_composition...")
        grade_rmse = compute_per_grade_rmse(
            y_true, preds_store["xgboost_composition"], df_test["BOF Grade Code"]
        )
        grade_rmse.to_csv(os.path.join(REPORTS_DIR, "per_grade_rmse.csv"), index=False)
        print(f"  Saved → reports/per_grade_rmse.csv")
        print(f"\n  Top 10 grades by heat count:")
        print(grade_rmse.head(10).to_string(index=False))

    # Feature importance plots
    print("\nGenerating feature importance plots...")
    for mname in ["xgboost_composition", "random_forest"]:
        if mname in preds_store:
            model = joblib.load(os.path.join(MODELS_DIR, f"{mname}.pkl"))
            plot_feature_importance(model, feature_names, mname)

    print("\n" + "=" * 60)
    print("  Evaluation complete. Reports saved to reports/")
    print("=" * 60)


if __name__ == "__main__":
    main()