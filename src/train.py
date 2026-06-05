"""
train.py — FerroMind Model Training
SAIL Durgapur Steel Plant | Steel Melting Shop

Trains 4 multi-output regressors on sms_train.csv and saves
all models + scaler to models/.

Usage:
    python src/train.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
TRAIN_CSV  = os.path.join("data", "processed", "sms_train.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# EXACT column names from your CSVs
# ─────────────────────────────────────────────
TARGET_COLS = ["Mn", "Si", "C", "S", "P"]

FEATURE_COLS = [
    # Raw process inputs
    "FEMN Into Ladle",
    "SIMN Into Ladle",
    "FESI Into Ladle",
    "Hot Metal Addn. Weight",
    "Scrap Addn. Weight",
    "Lime Into Converter",
    "Dolomite Into Converter",
    "Lime Into Ladle",
    "Iron Ore Into Converter",
    "Measered O2 Mainblow II",
    "Measered O2 2nd Blow",
    "Duration of Mainblow I/II",
    # Hot metal chemistry
    "HM_P", "HM_S", "HM_Si",
    # Grade encoding
    "Grade_Encoded",
    # Engineered features
    "Total_FA_Ladle", "HM_Ratio", "O2_per_ton_HM",
    "Lime_Dolo_Ratio", "SiMn_FA_Fraction",
]

# C is log-transformed before training (skewness = 5.3)
LOG_TRANSFORM_COL = "C"
LOG_TRANSFORM_IDX = TARGET_COLS.index(LOG_TRANSFORM_COL)


# ─────────────────────────────────────────────
# LOAD & PREPARE
# ─────────────────────────────────────────────
def load_train_data():
    print(f"Loading {TRAIN_CSV} ...")
    df = pd.read_csv(TRAIN_CSV)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"  [WARN] Missing feature cols (will be skipped): {missing}")

    X = df[available_features].copy()
    y = df[TARGET_COLS].copy()

    # Fill any NaNs with column median
    X = X.fillna(X.median(numeric_only=True))

    print(f"  Train rows:    {len(X):,}")
    print(f"  Features used: {len(available_features)}")
    print(f"  Targets:       {TARGET_COLS}")
    return X, y, available_features


def apply_log_transform(y: pd.DataFrame) -> pd.DataFrame:
    y = y.copy()
    y[LOG_TRANSFORM_COL] = np.log1p(y[LOG_TRANSFORM_COL])
    return y


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
def get_models():
    return {
        "linear_regression": MultiOutputRegressor(
            LinearRegression(), n_jobs=-1,
        ),
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300, max_depth=12,
                min_samples_leaf=5, n_jobs=-1, random_state=42,
            ), n_jobs=1,
        ),
        "xgboost_composition": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=42, verbosity=0,
            ), n_jobs=1,
        ),
        "lightgbm_composition": MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=42, verbosity=-1,
            ), n_jobs=1,
        ),
    }


# ─────────────────────────────────────────────
# TRAIN & SAVE
# ─────────────────────────────────────────────
def train_and_save(X_scaled, y_log, feature_names):
    models = get_models()
    for name, model in models.items():
        print(f"\n  Training {name} ...")
        model.fit(X_scaled, y_log.values)
        save_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, save_path)
        print(f"  Saved → {save_path}")
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FerroMind — Model Training")
    print("=" * 60)

    X, y, feature_names = load_train_data()

    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"  Saved → models/scaler.pkl")

    print(f"\nApplying log1p transform to '{LOG_TRANSFORM_COL}'...")
    y_log = apply_log_transform(y)

    print("\nTraining models...")
    train_and_save(X_scaled, y_log, feature_names)

    print("\n" + "=" * 60)
    print("  Training complete. Models saved to models/")
    print("  Primary model : xgboost_composition.pkl")
    print("  Scaler        : scaler.pkl")
    print("  Feature list  : feature_names.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()