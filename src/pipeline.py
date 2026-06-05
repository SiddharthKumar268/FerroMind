"""
pipeline.py — FerroMind Data Pipeline
SAIL Durgapur Steel Plant | Steel Melting Shop

Reads raw Excel files, cleans, merges, engineers features,
removes outliers, and writes chronological 80/20 train/test CSVs.

Usage:
    python pipeline.py
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

FILE_ADDITIONS   = os.path.join(RAW_DIR, "PRAVARTANAM-ADDITIONS_SMS.xlsx")
FILE_COMPOSITION = os.path.join(RAW_DIR, "Liquid_Steel_Composition.xlsx")
FILE_HM_ANALYSIS = os.path.join(RAW_DIR, "PRAVARTANAM-HM_ANALYSIS.xlsx")

OUT_TRAIN = os.path.join(PROCESSED_DIR, "sms_train.csv")
OUT_TEST  = os.path.join(PROCESSED_DIR, "sms_test.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DOMAIN-KNOWLEDGE RANGE FILTERS
# Values outside these ranges are physically
# impossible — flag as corrupt and drop the row.
# ─────────────────────────────────────────────
VALID_RANGES = {
    # Ferro-alloy additions (kg/heat)
    "FeMn_Ladle"    : (0,  2500),
    "SiMn_Ladle"    : (0,  3000),
    "FeSi_Ladle"    : (0,  1500),
    "FeMn_Furnace"  : (0,  2000),
    "SiMn_Furnace"  : (0,  2000),
    # O2 blow (Nm³/heat)
    "O2_Volume"     : (0, 12000),
    # Lime / Dolomite (kg/heat)
    "Lime"          : (0,  8000),
    "Dolomite"      : (0,  4000),
    # Hot metal ratio
    "HM_Wt"         : (50,  350),   # tonnes
    "Scrap_Wt"      : (0,   120),
    # Target chemistry (wt%)
    "Mn_Final"      : (0.01, 2.5),
    "Si_Final"      : (0.00, 1.2),
    "C_Final"       : (0.01, 1.2),
    "S_Final"       : (0.001, 0.08),
    "P_Final"       : (0.002, 0.06),
    # Hot metal chemistry (wt%)
    "HM_P"          : (0.05, 0.35),
    "HM_S"          : (0.01, 0.10),
    "HM_Si"         : (0.10, 1.50),
}

# ─────────────────────────────────────────────
# HELPER: concatenate all FY sheets in a file
# ─────────────────────────────────────────────
def load_all_sheets(filepath: str) -> pd.DataFrame:
    """
    Read every sheet in the workbook and vertically stack them.
    Sheets that cannot be parsed are skipped with a warning.
    """
    xl = pd.ExcelFile(filepath)
    frames = []
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
            df["_source_sheet"] = sheet
            frames.append(df)
        except Exception as e:
            print(f"  [WARN] Skipping sheet '{sheet}' in {os.path.basename(filepath)}: {e}")
    if not frames:
        raise ValueError(f"No readable sheets found in {filepath}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {os.path.basename(filepath)}: {len(combined):,} rows across {len(frames)} sheets")
    return combined


# ─────────────────────────────────────────────
# STEP 1 — Load raw files
# ─────────────────────────────────────────────
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n[1/6] Loading raw Excel files...")
    additions   = load_all_sheets(FILE_ADDITIONS)
    composition = load_all_sheets(FILE_COMPOSITION)
    hm_analysis = load_all_sheets(FILE_HM_ANALYSIS)
    return additions, composition, hm_analysis


# ─────────────────────────────────────────────
# STEP 2 — Standardise column names
# Map whatever the Excel headers are to the
# canonical names used throughout this pipeline.
# Extend this dict if the source files change.
# ─────────────────────────────────────────────
ADDITIONS_RENAME = {
    # additions file → canonical
    "Heat No"          : "Heat_No",
    "Heat Number"      : "Heat_No",
    "HEAT NO"          : "Heat_No",
    "Date"             : "Date",
    "Grade Code"       : "Grade_Code",
    "Grade"            : "Grade_Code",
    "BOF Grade"        : "Grade_Code",
    "FeMn Ladle"       : "FeMn_Ladle",
    "FEMN LADLE"       : "FeMn_Ladle",
    "SiMn Ladle"       : "SiMn_Ladle",
    "SIMN LADLE"       : "SiMn_Ladle",
    "FeSi Ladle"       : "FeSi_Ladle",
    "FESI LADLE"       : "FeSi_Ladle",
    "FeMn Furnace"     : "FeMn_Furnace",
    "FEMN FURNACE"     : "FeMn_Furnace",
    "SiMn Furnace"     : "SiMn_Furnace",
    "SIMN FURNACE"     : "SiMn_Furnace",
    "O2 Volume"        : "O2_Volume",
    "O2"               : "O2_Volume",
    "Lime"             : "Lime",
    "LIME"             : "Lime",
    "Dolomite"         : "Dolomite",
    "DOLOMITE"         : "Dolomite",
    "HM Wt"            : "HM_Wt",
    "HM WT"            : "HM_Wt",
    "Hot Metal Weight" : "HM_Wt",
    "Scrap Wt"         : "Scrap_Wt",
    "SCRAP WT"         : "Scrap_Wt",
    "Tap Weight"       : "Tap_Wt",
    "TAP WT"           : "Tap_Wt",
    "Tap Wt"           : "Tap_Wt",
}

COMPOSITION_RENAME = {
    "Heat No"      : "Heat_No",
    "Heat Number"  : "Heat_No",
    "HEAT NO"      : "Heat_No",
    "Date"         : "Date",
    "Mn"           : "Mn_Final",
    "MN"           : "Mn_Final",
    "Si"           : "Si_Final",
    "SI"           : "Si_Final",
    "C"            : "C_Final",
    "S"            : "S_Final",
    "P"            : "P_Final",
}

HM_RENAME = {
    "Date"  : "Date",
    "P"     : "HM_P",
    "S"     : "HM_S",
    "Si"    : "HM_Si",
    "SI"    : "HM_Si",
    "Mn"    : "HM_Mn",
}


def standardise(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns using map; silently skip unmapped columns."""
    present = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=present)


# ─────────────────────────────────────────────
# STEP 3 — Clean each source table
# ─────────────────────────────────────────────
def clean_additions(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise(df, ADDITIONS_RENAME)

    # Drop rows with no Heat_No
    df = df.dropna(subset=["Heat_No"])
    df["Heat_No"] = df["Heat_No"].astype(str).str.strip()

    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # FA nulls → 0 (not added = zero, not missing)
    fa_cols = ["FeMn_Ladle", "SiMn_Ladle", "FeSi_Ladle", "FeMn_Furnace", "SiMn_Furnace"]
    for col in fa_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Other numeric cols: coerce
    num_cols = ["O2_Volume", "Lime", "Dolomite", "HM_Wt", "Scrap_Wt", "Tap_Wt"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply range filters
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            bad = ~df[col].between(lo, hi) & df[col].notna()
            df.loc[bad, col] = np.nan   # nullify corrupt; don't drop yet

    # Grade Code: strip whitespace, upper
    if "Grade_Code" in df.columns:
        df["Grade_Code"] = df["Grade_Code"].astype(str).str.strip().str.upper()

    df = df.drop(columns=["_source_sheet"], errors="ignore")
    df = df.drop_duplicates(subset=["Heat_No"])
    return df


def clean_composition(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise(df, COMPOSITION_RENAME)
    df = df.dropna(subset=["Heat_No"])
    df["Heat_No"] = df["Heat_No"].astype(str).str.strip()

    target_cols = ["Mn_Final", "Si_Final", "C_Final", "S_Final", "P_Final"]
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply range filters on targets
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            bad = ~df[col].between(lo, hi) & df[col].notna()
            df.loc[bad, col] = np.nan

    df = df.drop(columns=["_source_sheet"], errors="ignore")
    df = df.drop_duplicates(subset=["Heat_No"])
    # Keep rows only where all 5 targets are present
    df = df.dropna(subset=target_cols)
    return df


def clean_hm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise(df, HM_RENAME)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    hm_cols = ["HM_P", "HM_S", "HM_Si"]
    for col in hm_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            bad = ~df[col].between(lo, hi) & df[col].notna()
            df.loc[bad, col] = np.nan

    # Daily average — one row per date
    hm_present = [c for c in hm_cols if c in df.columns]
    df = df.groupby("Date")[hm_present].mean().reset_index()
    df = df.drop(columns=["_source_sheet"], errors="ignore")
    return df


# ─────────────────────────────────────────────
# STEP 4 — Merge
# ─────────────────────────────────────────────
def merge_tables(
    additions: pd.DataFrame,
    composition: pd.DataFrame,
    hm_analysis: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[3/6] Merging tables...")

    # Inner join on Heat_No
    merged = pd.merge(additions, composition[
        ["Heat_No", "Mn_Final", "Si_Final", "C_Final", "S_Final", "P_Final"]
    ], on="Heat_No", how="inner")
    print(f"  After Additions ⋈ Composition: {len(merged):,} rows")

    # Left join HM daily average by date
    merged = pd.merge(merged, hm_analysis, on="Date", how="left")
    print(f"  After joining HM Analysis: {len(merged):,} rows")

    return merged


# ─────────────────────────────────────────────
# STEP 5 — Feature engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/6] Engineering features...")

    # Total FA added at ladle stage (FeMn + SiMn + FeSi)
    fa_ladle_cols = [c for c in ["FeMn_Ladle", "SiMn_Ladle", "FeSi_Ladle"] if c in df.columns]
    df["Total_FA_Ladle"] = df[fa_ladle_cols].sum(axis=1)

    # HM Ratio = HM_Wt / (HM_Wt + Scrap_Wt) — proxy for carbon input
    if "HM_Wt" in df.columns and "Scrap_Wt" in df.columns:
        total_charge = df["HM_Wt"] + df["Scrap_Wt"].fillna(0)
        df["HM_Ratio"] = df["HM_Wt"] / total_charge.replace(0, np.nan)
    else:
        df["HM_Ratio"] = np.nan

    # O2 per tonne of hot metal — normalises blow intensity
    if "O2_Volume" in df.columns and "HM_Wt" in df.columns:
        df["O2_per_ton_HM"] = df["O2_Volume"] / df["HM_Wt"].replace(0, np.nan)
    else:
        df["O2_per_ton_HM"] = np.nan

    # Lime to Dolomite ratio — flux balance proxy
    if "Lime" in df.columns and "Dolomite" in df.columns:
        df["Lime_Dolo_Ratio"] = df["Lime"] / (df["Dolomite"].replace(0, np.nan))
    else:
        df["Lime_Dolo_Ratio"] = np.nan

    # SiMn fraction of total FA at ladle
    if "SiMn_Ladle" in df.columns:
        df["SiMn_FA_Fraction"] = df["SiMn_Ladle"] / df["Total_FA_Ladle"].replace(0, np.nan)
    else:
        df["SiMn_FA_Fraction"] = np.nan

    # Label-encode Grade_Code — essential for Si, C, P accuracy
    if "Grade_Code" in df.columns:
        le = LabelEncoder()
        df["Grade_Code_Enc"] = le.fit_transform(df["Grade_Code"].fillna("UNKNOWN"))
        # Save mapping for reference
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"  Grade codes encoded: {len(mapping)} distinct grades")
    else:
        df["Grade_Code_Enc"] = 0

    print(f"  Engineered features: Total_FA_Ladle, HM_Ratio, O2_per_ton_HM, "
          f"Lime_Dolo_Ratio, SiMn_FA_Fraction, Grade_Code_Enc")
    return df


# ─────────────────────────────────────────────
# STEP 6 — Remove outliers (1st–99th pct on targets)
# ─────────────────────────────────────────────
TARGET_COLS = ["Mn_Final", "Si_Final", "C_Final", "S_Final", "P_Final"]

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/6] Removing outliers on target variables...")
    before = len(df)
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df = df[df[col].between(lo, hi)]
    after = len(df)
    print(f"  Rows removed: {before - after:,}  ({100*(before-after)/before:.1f}%)")
    print(f"  Rows retained: {after:,}")
    return df


# ─────────────────────────────────────────────
# STEP 7 — Chronological 80/20 split
# ─────────────────────────────────────────────
def split_chronological(df: pd.DataFrame, train_frac: float = 0.80):
    df = df.sort_values("Date").reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    train = df.iloc[:cutoff].copy()
    test  = df.iloc[cutoff:].copy()
    print(f"\n[6/6] Chronological split:")
    print(f"  Train: {len(train):,} rows  ({train['Date'].min().date()} → {train['Date'].max().date()})")
    print(f"  Test:  {len(test):,} rows  ({test['Date'].min().date()} → {test['Date'].max().date()})")
    return train, test


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FerroMind — Data Pipeline")
    print("=" * 60)

    # 1. Load
    additions_raw, composition_raw, hm_raw = load_raw()

    # 2. Clean
    print("\n[2/6] Cleaning tables...")
    additions   = clean_additions(additions_raw)
    composition = clean_composition(composition_raw)
    hm_analysis = clean_hm_analysis(hm_raw)
    print(f"  Additions clean:   {len(additions):,} rows")
    print(f"  Composition clean: {len(composition):,} rows")
    print(f"  HM Analysis clean: {len(hm_analysis):,} days")

    # 3. Merge
    merged = merge_tables(additions, composition, hm_analysis)

    # 4. Feature engineering
    merged = engineer_features(merged)

    # 5. Outlier removal
    merged = remove_outliers(merged)

    # 6. Split
    train, test = split_chronological(merged)

    # 7. Save
    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST,  index=False)
    print(f"\n  Saved → {OUT_TRAIN}")
    print(f"  Saved → {OUT_TEST}")

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Total heats:   {len(merged):,}")
    print(f"  Features:      {merged.shape[1] - len(TARGET_COLS)} columns")
    print(f"  Target cols:   {TARGET_COLS}")
    print("=" * 60)


if __name__ == "__main__":
    main()