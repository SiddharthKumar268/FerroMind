import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import itertools
import sys

MODELS_DIR  = "models"
MODEL_PATH  = os.path.join(MODELS_DIR, "xgboost_composition.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEAT_PATH   = os.path.join(MODELS_DIR, "feature_names.pkl")

TARGET_COLS       = ["Mn", "Si", "C", "S", "P"]
LOG_TRANSFORM_IDX = TARGET_COLS.index("C")
FA_COLS = ["FEMN Into Ladle", "SIMN Into Ladle", "FESI Into Ladle"]

FA_BOUNDS = {
    "FEMN Into Ladle" : (0, 2500),
    "SIMN Into Ladle" : (0, 3000),
    "FESI Into Ladle" : (0, 1500),
}

SAMPLE_HEAT = {
    "FEMN Into Ladle"          : 400,
    "SIMN Into Ladle"          : 1674,
    "FESI Into Ladle"          : 260,
    "Hot Metal Addn. Weight"   : 131.81,
    "Scrap Addn. Weight"       : 0.0,
    "Lime Into Converter"      : 9765,
    "Dolomite Into Converter"  : 1965,
    "Lime Into Ladle"          : 803,
    "Iron Ore Into Converter"  : 5150,
    "Measered O2 Mainblow II"  : 6594,
    "Measered O2 2nd Blow"     : 0,
    "Duration of Mainblow I/II": 994,
    "HM_P"                     : 0.143,
    "HM_S"                     : 0.043,
    "HM_Si"                    : 0.861,
    "Grade_Encoded"            : 43,
}

SAMPLE_TARGET = {
    "Mn": [0.60, 0.80],
    "Si": [0.15, 0.30],
    "C" : [0.10, 0.22],
    "S" : [0.00, 0.035],
    "P" : [0.00, 0.025],
}

def load_artifacts():
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEAT_PATH)
    return model, scaler, feature_names

def prepare_input_batch(heats_list, feature_names):
    df = pd.DataFrame(heats_list)
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0 if feat in FA_COLS else np.nan

    femn  = df["FEMN Into Ladle"].fillna(0)
    simn  = df["SIMN Into Ladle"].fillna(0)
    fesi  = df["FESI Into Ladle"].fillna(0)
    hm_wt = df["Hot Metal Addn. Weight"]
    scrap = df["Scrap Addn. Weight"].fillna(0)
    o2    = df["Measered O2 Mainblow II"]
    lime  = df["Lime Into Converter"]
    dolo  = df["Dolomite Into Converter"]

    total_fa = femn + simn + fesi
    df["Total_FA_Ladle"]   = total_fa
    df["HM_Ratio"]         = hm_wt / (hm_wt + scrap)
    df["O2_per_ton_HM"]    = o2 / hm_wt
    df["Lime_Dolo_Ratio"]  = lime / dolo
    df["SiMn_FA_Fraction"] = simn / total_fa
    
    df["SiMn_FA_Fraction"] = df["SiMn_FA_Fraction"].fillna(0)
    df = df.fillna(0)
    df = df[feature_names]
    return df

def predict_batch(heats_list, model, scaler, feature_names):
    df = prepare_input_batch(heats_list, feature_names)
    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    preds[:, LOG_TRANSFORM_IDX] = np.expm1(preds[:, LOG_TRANSFORM_IDX])
    return preds

def optimise_fa_vectorized(heat_dict, target_range, model, scaler, feature_names, step=50):
    femn_vals = list(range(0, FA_BOUNDS["FEMN Into Ladle"][1] + step, step))
    simn_vals = list(range(0, FA_BOUNDS["SIMN Into Ladle"][1] + step, step))
    fesi_vals = list(range(0, FA_BOUNDS["FESI Into Ladle"][1] + step, step))

    candidates = []
    fa_totals  = []
    for femn, simn, fesi in itertools.product(femn_vals, simn_vals, fesi_vals):
        candidate = {
            **heat_dict,
            "FEMN Into Ladle": femn,
            "SIMN Into Ladle": simn,
            "FESI Into Ladle": fesi,
        }
        candidates.append(candidate)
        fa_totals.append(femn + simn + fesi)

    preds = predict_batch(candidates, model, scaler, feature_names)

    target_elems = list(target_range.keys())
    target_idxs  = [TARGET_COLS.index(e) for e in target_elems if e in TARGET_COLS]
    target_lo    = np.array([target_range[TARGET_COLS[i]][0] for i in target_idxs])
    target_hi    = np.array([target_range[TARGET_COLS[i]][1] for i in target_idxs])

    pred_targets = preds[:, target_idxs]
    lo_violations = np.maximum(target_lo - pred_targets, 0)
    hi_violations = np.maximum(pred_targets - target_hi, 0)
    total_violations = np.sum(lo_violations + hi_violations, axis=1)
    feasible_mask = total_violations == 0
    fa_arr = np.array(fa_totals, dtype=float)

    if np.any(feasible_mask):
        feasible_fa = fa_arr.copy()
        feasible_fa[~feasible_mask] = float("inf")
        best_idx = int(np.argmin(feasible_fa))
        c = candidates[best_idx]
        best_preds = {col: round(float(preds[best_idx, i]), 5) for i, col in enumerate(TARGET_COLS)}
        return {
            "FEMN Into Ladle": c["FEMN Into Ladle"],
            "SIMN Into Ladle": c["SIMN Into Ladle"],
            "FESI Into Ladle": c["FESI Into Ladle"],
            "Total_FA_Ladle" : int(fa_arr[best_idx]),
            "predicted"      : best_preds,
            "feasible"       : True,
        }
    else:
        best_idx = int(np.argmin(total_violations))
        c = candidates[best_idx]
        best_preds = {col: round(float(preds[best_idx, i]), 5) for i, col in enumerate(TARGET_COLS)}
        return {
            "FEMN Into Ladle" : c["FEMN Into Ladle"],
            "SIMN Into Ladle" : c["SIMN Into Ladle"],
            "FESI Into Ladle" : c["FESI Into Ladle"],
            "Total_FA_Ladle"  : int(fa_arr[best_idx]),
            "predicted"       : best_preds,
            "feasible"        : False,
            "total_violation" : round(float(total_violations[best_idx]), 5),
        }

def optimise_fa_hierarchical(heat_dict, target_range, model, scaler, feature_names, step=1, coarse_step=50):
    # Stage 1: Coarse search
    res1 = optimise_fa_vectorized(heat_dict, target_range, model, scaler, feature_names, step=coarse_step)
    if step >= coarse_step:
        return res1

    # Stage 2: Medium search around res1
    femn_c, simn_c, fesi_c = res1["FEMN Into Ladle"], res1["SIMN Into Ladle"], res1["FESI Into Ladle"]
    medium_step = 10
    femn_range = range(max(0, femn_c - coarse_step), min(FA_BOUNDS["FEMN Into Ladle"][1], femn_c + coarse_step) + medium_step, medium_step)
    simn_range = range(max(0, simn_c - coarse_step), min(FA_BOUNDS["SIMN Into Ladle"][1], simn_c + coarse_step) + medium_step, medium_step)
    fesi_range = range(max(0, fesi_c - coarse_step), min(FA_BOUNDS["FESI Into Ladle"][1], fesi_c + coarse_step) + medium_step, medium_step)

    candidates = []
    fa_totals = []
    for femn, simn, fesi in itertools.product(femn_range, simn_range, fesi_range):
        candidate = {
            **heat_dict,
            "FEMN Into Ladle": femn,
            "SIMN Into Ladle": simn,
            "FESI Into Ladle": fesi,
        }
        candidates.append(candidate)
        fa_totals.append(femn + simn + fesi)

    preds = predict_batch(candidates, model, scaler, feature_names)

    target_elems = list(target_range.keys())
    target_idxs  = [TARGET_COLS.index(e) for e in target_elems if e in TARGET_COLS]
    target_lo    = np.array([target_range[TARGET_COLS[i]][0] for i in target_idxs])
    target_hi    = np.array([target_range[TARGET_COLS[i]][1] for i in target_idxs])

    pred_targets = preds[:, target_idxs]
    lo_violations = np.maximum(target_lo - pred_targets, 0)
    hi_violations = np.maximum(pred_targets - target_hi, 0)
    total_violations = np.sum(lo_violations + hi_violations, axis=1)
    feasible_mask = total_violations == 0
    fa_arr = np.array(fa_totals, dtype=float)

    res2 = None
    if np.any(feasible_mask):
        feasible_fa = fa_arr.copy()
        feasible_fa[~feasible_mask] = float("inf")
        best_idx = int(np.argmin(feasible_fa))
        c = candidates[best_idx]
        best_preds = {col: round(float(preds[best_idx, i]), 5) for i, col in enumerate(TARGET_COLS)}
        res2 = {
            "FEMN Into Ladle": c["FEMN Into Ladle"],
            "SIMN Into Ladle": c["SIMN Into Ladle"],
            "FESI Into Ladle": c["FESI Into Ladle"],
            "Total_FA_Ladle" : int(fa_arr[best_idx]),
            "predicted"      : best_preds,
            "feasible"       : True,
        }
    else:
        best_idx = int(np.argmin(total_violations))
        c = candidates[best_idx]
        best_preds = {col: round(float(preds[best_idx, i]), 5) for i, col in enumerate(TARGET_COLS)}
        res2 = {
            "FEMN Into Ladle" : c["FEMN Into Ladle"],
            "SIMN Into Ladle" : c["SIMN Into Ladle"],
            "FESI Into Ladle" : c["FESI Into Ladle"],
            "Total_FA_Ladle"  : int(fa_arr[best_idx]),
            "predicted"       : best_preds,
            "feasible"        : False,
            "total_violation" : round(float(total_violations[best_idx]), 5),
        }

    if step >= 10:
        return res2

    # Stage 3: Fine search
    femn_m, simn_m, fesi_m = res2["FEMN Into Ladle"], res2["SIMN Into Ladle"], res2["FESI Into Ladle"]
    femn_range = range(max(0, femn_m - 10), min(FA_BOUNDS["FEMN Into Ladle"][1], femn_m + 10) + step, step)
    simn_range = range(max(0, simn_m - 10), min(FA_BOUNDS["SIMN Into Ladle"][1], simn_m + 10) + step, step)
    fesi_range = range(max(0, fesi_m - 10), min(FA_BOUNDS["FESI Into Ladle"][1], fesi_m + 10) + step, step)

    candidates = []
    fa_totals = []
    for femn, simn, fesi in itertools.product(femn_range, simn_range, fesi_range):
        candidate = {
            **heat_dict,
            "FEMN Into Ladle": femn,
            "SIMN Into Ladle": simn,
            "FESI Into Ladle": fesi,
        }
        candidates.append(candidate)
        fa_totals.append(femn + simn + fesi)

    preds = predict_batch(candidates, model, scaler, feature_names)
    pred_targets = preds[:, target_idxs]
    lo_violations = np.maximum(target_lo - pred_targets, 0)
    hi_violations = np.maximum(pred_targets - target_hi, 0)
    total_violations = np.sum(lo_violations + hi_violations, axis=1)
    feasible_mask = total_violations == 0
    fa_arr = np.array(fa_totals, dtype=float)

    if np.any(feasible_mask):
        feasible_fa = fa_arr.copy()
        feasible_fa[~feasible_mask] = float("inf")
        best_idx = int(np.argmin(feasible_fa))
        c = candidates[best_idx]
        best_preds = {col: round(float(preds[best_idx, i]), 5) for i, col in enumerate(TARGET_COLS)}
        return {
            "FEMN Into Ladle": c["FEMN Into Ladle"],
            "SIMN Into Ladle": c["SIMN Into Ladle"],
            "FESI Into Ladle": c["FESI Into Ladle"],
            "Total_FA_Ladle" : int(fa_arr[best_idx]),
            "predicted"      : best_preds,
            "feasible"       : True,
        }
    else:
        best_idx = int(np.argmin(total_violations))
        c = candidates[best_idx]
        best_preds = {col: round(float(preds[best_idx, i]), 5) for i, col in enumerate(TARGET_COLS)}
        return {
            "FEMN Into Ladle" : c["FEMN Into Ladle"],
            "SIMN Into Ladle" : c["SIMN Into Ladle"],
            "FESI Into Ladle" : c["FESI Into Ladle"],
            "Total_FA_Ladle"  : int(fa_arr[best_idx]),
            "predicted"       : best_preds,
            "feasible"        : False,
            "total_violation" : round(float(total_violations[best_idx]), 5),
        }

def main():
    model, scaler, feature_names = load_artifacts()
    
    out_file = os.path.join("src", "opt_results.txt")
    with open(out_file, "w") as f:
        f.write("Evaluating Hierarchical Optimization across step sizes and coarse stages...\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Coarse Step':<11} | {'Step (kg)':<9} | {'FEMN':<6} | {'SIMN':<6} | {'FESI':<6} | {'Total FA':<9} | {'Feasible':<8} | {'Time (ms)':<9}\n")
        f.write("-" * 120 + "\n")

        for coarse in [50, 100]:
            for step in [50, 25, 20, 10, 5, 2, 1]:
                t0 = time.perf_counter()
                res_h = optimise_fa_hierarchical(SAMPLE_HEAT, SAMPLE_TARGET, model, scaler, feature_names, step, coarse_step=coarse)
                t1 = time.perf_counter()
                dt_ms_h = (t1 - t0) * 1000
                f.write(f"{coarse:<11} | {step:<9} | {res_h['FEMN Into Ladle']:<6} | {res_h['SIMN Into Ladle']:<6} | {res_h['FESI Into Ladle']:<6} | {res_h['Total_FA_Ladle']:<9} | {str(res_h['feasible']):<8} | {dt_ms_h:<9.2f}\n")
            
        f.write("-" * 120 + "\n")
    print(f"Results written to {out_file}")

if __name__ == "__main__":
    main()
