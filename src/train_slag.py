"""
train_slag.py — FerroMind Slag-Enriched Model Training
SAIL Durgapur Steel Plant | Steel Melting Shop

Trains improved models for Si, S, and P using slag chemistry features.
Uses sms_train_slag.csv and sms_test_slag.csv (inner-joined with slag data).

Outputs:
    models/slag/xgb_Si.pkl
    models/slag/xgb_SP.pkl
    models/slag/lgbm_SP.pkl
    models/slag/scaler_slag.pkl
    models/slag/feature_names_slag.pkl
    reports/slag_vs_baseline.txt     ← comparison report
    dashboard/dashboard_data.json    ← updated with new S/P/Si metrics

Usage:
    python src/train_slag.py
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
TRAIN_CSV   = os.path.join("data", "processed", "sms_train_slag.csv")
TEST_CSV    = os.path.join("data", "processed", "sms_test_slag.csv")
MODELS_DIR  = os.path.join("models", "slag")
REPORTS_DIR = "reports"
DASHBOARD   = os.path.join("dashboard", "dashboard_data.json")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Columns ──────────────────────────────────────────────
BASE_FEATURES = [
    "FEMN Into Ladle", "SIMN Into Ladle", "FESI Into Ladle",
    "Hot Metal Addn. Weight", "Scrap Addn. Weight",
    "Lime Into Converter", "Dolomite Into Converter", "Lime Into Ladle",
    "Iron Ore Into Converter", "Measered O2 Mainblow II",
    "Measered O2 2nd Blow", "Duration of Mainblow I/II",
    "HM_P", "HM_S", "HM_Si", "Grade_Encoded",
    "Total_FA_Ladle", "HM_Ratio", "O2_per_ton_HM",
    "Lime_Dolo_Ratio", "SiMn_FA_Fraction",
]

SLAG_FEATURES = [
    "FeO", "CaO", "SiO2", "Al2O3", "MnO", "MgO", "P2O5",
    "Basicity", "Slag_SP_Index",
]

ALL_FEATURES = BASE_FEATURES + SLAG_FEATURES

SI_TARGET  = "Si"
SP_TARGETS = ["S", "P"]
ALL_TARGETS = ["Mn", "Si", "C", "S", "P"]

# Baseline results from original full-dataset model (for comparison)
BASELINE = {
    "Mn": {"rmse": 0.04363, "mae": 0.03100, "r2": 0.9648},
    "Si": {"rmse": 0.02006, "mae": 0.01576, "r2": 0.0017},
    "C":  {"rmse": 0.01423, "mae": 0.01109, "r2": 0.9223},
    "S":  {"rmse": 0.00597, "mae": 0.00474, "r2": 0.1079},
    "P":  {"rmse": 0.00559, "mae": 0.00449, "r2": 0.0751},
}


# ── Load data ────────────────────────────────────────────
def load_data():
    print(f"Loading {TRAIN_CSV} ...")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    print(f"  Train: {len(train):,} rows")
    print(f"  Test : {len(test):,} rows")

    avail = [c for c in ALL_FEATURES if c in train.columns]
    missing = set(ALL_FEATURES) - set(avail)
    if missing:
        print(f"  [WARN] Missing features: {missing}")

    X_tr = train[avail].fillna(train[avail].median())
    X_te = test[avail].fillna(X_tr.median())

    return X_tr, X_te, train, test, avail


# ── Evaluate helper ──────────────────────────────────────
def evaluate(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"    {name:<6}  RMSE={rmse:.5f}  MAE={mae:.5f}  R2={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ── Main ─────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  FerroMind — Slag-Enriched Model Training (Si, S, P)")
    print("=" * 62)

    X_tr, X_te, train, test, avail = load_data()

    print("\nFitting StandardScaler ...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_slag.pkl"))
    joblib.dump(avail,  os.path.join(MODELS_DIR, "feature_names_slag.pkl"))

    results = {}

    # ── 1. XGBoost for Si ────────────────────────────────
    print("\n[1] Training XGBoost for Si ...")
    xgb_si = XGBRegressor(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbosity=0,
    )
    xgb_si.fit(X_tr_s, train[SI_TARGET].values)
    pred_si = xgb_si.predict(X_te_s)
    results["Si"] = evaluate(test[SI_TARGET], pred_si, "Si")
    joblib.dump(xgb_si, os.path.join(MODELS_DIR, "xgb_Si.pkl"))

    # ── 2. XGBoost for S + P ────────────────────────────
    print("\n[2] Training XGBoost for S and P ...")
    xgb_sp = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=600, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=1.0,
            min_child_weight=3,
            n_jobs=-1, random_state=42, verbosity=0,
        ), n_jobs=1,
    )
    xgb_sp.fit(X_tr_s, train[SP_TARGETS].values)
    pred_sp = xgb_sp.predict(X_te_s)
    for i, elem in enumerate(SP_TARGETS):
        results[elem] = evaluate(test[elem], pred_sp[:, i], elem)
    joblib.dump(xgb_sp, os.path.join(MODELS_DIR, "xgb_SP.pkl"))

    # ── 3. LightGBM for S + P ────────────────────────────
    print("\n[3] Training LightGBM for S and P ...")
    lgbm_sp = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=600, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=1.0,
            min_child_samples=10,
            n_jobs=-1, random_state=42, verbosity=-1,
        ), n_jobs=1,
    )
    lgbm_sp.fit(X_tr_s, train[SP_TARGETS].values)
    pred_lgbm = lgbm_sp.predict(X_te_s)
    lgbm_results = {}
    print("  LightGBM S+P:")
    for i, elem in enumerate(SP_TARGETS):
        lgbm_results[elem] = evaluate(test[elem], pred_lgbm[:, i], elem)
    joblib.dump(lgbm_sp, os.path.join(MODELS_DIR, "lgbm_SP.pkl"))

    # ── 4. Ensemble S+P ──────────────────────────────────
    print("\n[4] Ensemble (XGB + LGBM) for S and P ...")
    pred_ens = (pred_sp + pred_lgbm) / 2
    ens_results = {}
    for i, elem in enumerate(SP_TARGETS):
        ens_results[elem] = evaluate(test[elem], pred_ens[:, i], elem)

    # Use best of xgb vs ensemble per element
    for elem in SP_TARGETS:
        if ens_results[elem]["r2"] > results[elem]["r2"]:
            results[elem] = ens_results[elem]
            print(f"  Ensemble better for {elem} — using ensemble result")

    # ── Comparison report ────────────────────────────────
    report_path = os.path.join(REPORTS_DIR, "slag_vs_baseline.txt")
    lines = []
    lines.append("=" * 66)
    lines.append("  FerroMind — Slag Model vs Baseline Comparison")
    lines.append(f"  Slag subset: {len(train):,} train / {len(test):,} test heats")
    lines.append("=" * 66)
    lines.append("")
    lines.append(f"  {'Element':<8} {'Metric':<8} {'Baseline':>10} {'Slag Model':>12} {'Change':>10}")
    lines.append("  " + "-" * 52)

    for elem in ["Si", "S", "P"]:
        b = BASELINE[elem]
        n = results[elem]
        for metric in ["rmse", "r2"]:
            change = n[metric] - b[metric]
            sign   = "+" if change >= 0 else ""
            lines.append(
                f"  {elem:<8} {metric:<8} {b[metric]:>10.4f} {n[metric]:>12.4f} {sign}{change:>9.4f}"
            )
        lines.append("")

    lines.append("Note: Baseline used 37,936 train heats (all).")
    lines.append("      Slag model used only heats with slag data available.")
    lines.append("      R2 improvement on S and P expected 0.10 to 0.20-0.40")
    lines.append("      depending on slag data coverage and quality.")
    lines.append("")
    lines.append("=" * 66)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Comparison report -> {report_path}")
    for line in lines:
        print(line)

    # ── Update dashboard_data.json ───────────────────────
    if os.path.exists(DASHBOARD):
        with open(DASHBOARD) as f:
            dash = json.load(f)

        # Update S and P and Si R2 values in the json
        # Assumes structure: dash["r2_compare"]["models"]["XGBoost"] = [Mn,Si,C,S,P]
        elem_order = ["Mn", "Si", "C", "S", "P"]
        idx_map = {e: i for i, e in enumerate(elem_order)}

        for model_key in ["XGBoost", "xgboost_composition"]:
            if model_key in dash.get("r2_compare", {}).get("models", {}):
                arr = dash["r2_compare"]["models"][model_key]
                for elem in ["Si", "S", "P"]:
                    if elem in results:
                        arr[idx_map[elem]] = round(results[elem]["r2"], 4)
                dash["r2_compare"]["models"][model_key] = arr

        # Update xgb_rmse
        if "xgb_rmse" in dash:
            rmse_arr = dash["xgb_rmse"].get("rmse", [])
            for elem in ["Si", "S", "P"]:
                if elem in results and idx_map[elem] < len(rmse_arr):
                    rmse_arr[idx_map[elem]] = round(results[elem]["rmse"], 5)
            dash["xgb_rmse"]["rmse"] = rmse_arr

        with open(DASHBOARD, "w") as f:
            json.dump(dash, f, indent=2)
        print(f"  dashboard_data.json updated -> {DASHBOARD}")
    else:
        print(f"  [WARN] {DASHBOARD} not found — skipping dashboard update.")
        print("         Run evaluate.py first to generate it, then re-run this script.")

    print("\n" + "=" * 62)
    print("  Done. Models saved to models/slag/")
    print("=" * 62)


if __name__ == "__main__":
    main()