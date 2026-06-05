"""
predict.py — FerroMind Inference & Optimisation (v2 — Performance)
SAIL Durgapur Steel Plant | Steel Melting Shop

Modes:
  predict       — given process inputs, return predicted final composition
  batch_predict — predict multiple heats in one vectorized call
  optimise      — given a target chemistry range, find minimum FA dose
                  (uses vectorized + hierarchical search)

Usage (CLI):
    python src/predict.py --mode predict
    python src/predict.py --mode optimise --step 50

Usage (REPL — used by Node.js server):
    Reads JSON lines from stdin, writes JSON responses to stdout.
    Stays alive until stdin closes.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import itertools

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
MODELS_DIR  = "models"
MODEL_PATH  = os.path.join(MODELS_DIR, "xgboost_composition.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEAT_PATH   = os.path.join(MODELS_DIR, "feature_names.pkl")

# ─────────────────────────────────────────────
# EXACT column names from your CSVs
# ─────────────────────────────────────────────
TARGET_COLS       = ["Mn", "Si", "C", "S", "P"]
LOG_TRANSFORM_IDX = TARGET_COLS.index("C")

# FA columns the optimiser varies
FA_COLS = ["FEMN Into Ladle", "SIMN Into Ladle", "FESI Into Ladle"]

FA_BOUNDS = {
    "FEMN Into Ladle" : (0, 2500),
    "SIMN Into Ladle" : (0, 3000),
    "FESI Into Ladle" : (0, 1500),
}


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
def load_artifacts():
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEAT_PATH)
    return model, scaler, feature_names


# ─────────────────────────────────────────────
# PREPARE INPUT ROW (single heat)
# ─────────────────────────────────────────────
def prepare_input(heat_dict, feature_names):
    row = {}
    for feat in feature_names:
        row[feat] = heat_dict.get(feat, 0 if feat in FA_COLS else np.nan)

    df = pd.DataFrame([row])

    # Recompute engineered features
    femn  = heat_dict.get("FEMN Into Ladle", 0) or 0
    simn  = heat_dict.get("SIMN Into Ladle", 0) or 0
    fesi  = heat_dict.get("FESI Into Ladle", 0) or 0
    hm_wt = heat_dict.get("Hot Metal Addn. Weight", np.nan)
    scrap = heat_dict.get("Scrap Addn. Weight", 0) or 0
    o2    = heat_dict.get("Measered O2 Mainblow II", np.nan)
    lime  = heat_dict.get("Lime Into Converter", np.nan)
    dolo  = heat_dict.get("Dolomite Into Converter", np.nan)

    total_fa = femn + simn + fesi
    df["Total_FA_Ladle"]   = total_fa
    df["HM_Ratio"]         = hm_wt / (hm_wt + scrap) if hm_wt else np.nan
    df["O2_per_ton_HM"]    = o2 / hm_wt if hm_wt and o2 else np.nan
    df["Lime_Dolo_Ratio"]  = lime / dolo if dolo else np.nan
    df["SiMn_FA_Fraction"] = simn / total_fa if total_fa else np.nan

    df = df.fillna(0)
    df = df[[f for f in feature_names if f in df.columns]]
    return df


# ─────────────────────────────────────────────
# PREPARE INPUT BATCH (multiple heats at once)
# ─────────────────────────────────────────────
def prepare_input_batch(heats_list, feature_names):
    df = pd.DataFrame(heats_list)
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0 if feat in FA_COLS else np.nan

    femn  = df["FEMN Into Ladle"].fillna(0)
    simn  = df["SIMN Into Ladle"].fillna(0)
    fesi  = df["FESI Into Ladle"].fillna(0)
    hm_wt = df.get("Hot Metal Addn. Weight", pd.Series(np.nan, index=df.index))
    scrap = df.get("Scrap Addn. Weight", pd.Series(0, index=df.index)).fillna(0)
    o2    = df.get("Measered O2 Mainblow II", pd.Series(np.nan, index=df.index))
    lime  = df.get("Lime Into Converter", pd.Series(np.nan, index=df.index))
    dolo  = df.get("Dolomite Into Converter", pd.Series(np.nan, index=df.index))

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


# ─────────────────────────────────────────────
# PREDICT (single)
# ─────────────────────────────────────────────
def predict_composition(heat_dict, model, scaler, feature_names):
    df     = prepare_input(heat_dict, feature_names)
    scaled = scaler.transform(df)
    preds  = model.predict(scaled)[0]
    preds[LOG_TRANSFORM_IDX] = np.expm1(preds[LOG_TRANSFORM_IDX])
    return {col: round(float(preds[i]), 5) for i, col in enumerate(TARGET_COLS)}


# ─────────────────────────────────────────────
# PREDICT BATCH (vectorized — much faster)
# ─────────────────────────────────────────────
def predict_batch(heats_list, model, scaler, feature_names):
    df = prepare_input_batch(heats_list, feature_names)
    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    preds[:, LOG_TRANSFORM_IDX] = np.expm1(preds[:, LOG_TRANSFORM_IDX])
    return preds


# ─────────────────────────────────────────────
# OPTIMISER — VECTORIZED (from test_opt.py)
# ─────────────────────────────────────────────
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
    else:
        best_idx = int(np.argmin(total_violations))

    c = candidates[best_idx]
    best_preds = {col: round(float(preds[best_idx, i]), 5) for i, col in enumerate(TARGET_COLS)}
    result = {
        "FEMN Into Ladle": c["FEMN Into Ladle"],
        "SIMN Into Ladle": c["SIMN Into Ladle"],
        "FESI Into Ladle": c["FESI Into Ladle"],
        "Total_FA_Ladle" : int(fa_arr[best_idx]),
        "predicted"      : best_preds,
        "feasible"       : bool(feasible_mask[best_idx]),
    }
    if not feasible_mask[best_idx]:
        result["total_violation"] = round(float(total_violations[best_idx]), 5)
    return result


# ─────────────────────────────────────────────
# OPTIMISER — HIERARCHICAL (coarse → fine)
# ─────────────────────────────────────────────
def optimise_fa_hierarchical(heat_dict, target_range, model, scaler, feature_names, step=50, coarse_step=100):
    # Stage 1: Coarse search
    res1 = optimise_fa_vectorized(heat_dict, target_range, model, scaler, feature_names, step=coarse_step)
    if step >= coarse_step:
        return res1

    # Stage 2: Refine around best result
    femn_c = res1["FEMN Into Ladle"]
    simn_c = res1["SIMN Into Ladle"]
    fesi_c = res1["FESI Into Ladle"]

    margin = coarse_step
    fine_step = max(step, 10)

    femn_range = range(max(0, femn_c - margin), min(FA_BOUNDS["FEMN Into Ladle"][1], femn_c + margin) + fine_step, fine_step)
    simn_range = range(max(0, simn_c - margin), min(FA_BOUNDS["SIMN Into Ladle"][1], simn_c + margin) + fine_step, fine_step)
    fesi_range = range(max(0, fesi_c - margin), min(FA_BOUNDS["FESI Into Ladle"][1], fesi_c + margin) + fine_step, fine_step)

    candidates = []
    fa_totals  = []
    for femn, simn, fesi in itertools.product(femn_range, simn_range, fesi_range):
        candidates.append({**heat_dict, "FEMN Into Ladle": femn, "SIMN Into Ladle": simn, "FESI Into Ladle": fesi})
        fa_totals.append(femn + simn + fesi)

    preds = predict_batch(candidates, model, scaler, feature_names)

    target_idxs = [TARGET_COLS.index(e) for e in target_range if e in TARGET_COLS]
    target_lo   = np.array([target_range[TARGET_COLS[i]][0] for i in target_idxs])
    target_hi   = np.array([target_range[TARGET_COLS[i]][1] for i in target_idxs])

    pred_targets = preds[:, target_idxs]
    lo_v = np.maximum(target_lo - pred_targets, 0)
    hi_v = np.maximum(pred_targets - target_hi, 0)
    total_v = np.sum(lo_v + hi_v, axis=1)
    feasible_mask = total_v == 0
    fa_arr = np.array(fa_totals, dtype=float)

    if np.any(feasible_mask):
        feasible_fa = fa_arr.copy()
        feasible_fa[~feasible_mask] = float("inf")
        best_idx = int(np.argmin(feasible_fa))
    else:
        best_idx = int(np.argmin(total_v))

    c = candidates[best_idx]
    best_preds = {col: round(float(preds[best_idx, i]), 5) for i, col in enumerate(TARGET_COLS)}
    result = {
        "FEMN Into Ladle": c["FEMN Into Ladle"],
        "SIMN Into Ladle": c["SIMN Into Ladle"],
        "FESI Into Ladle": c["FESI Into Ladle"],
        "Total_FA_Ladle" : int(fa_arr[best_idx]),
        "predicted"      : best_preds,
        "feasible"       : bool(feasible_mask[best_idx]),
    }
    if not feasible_mask[best_idx]:
        result["total_violation"] = round(float(total_v[best_idx]), 5)
    return result


# ─────────────────────────────────────────────
# SAMPLE DATA (used when no --input given)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# REPL MODE — persistent process for Node.js
# Reads JSON lines from stdin, responds on stdout
# ─────────────────────────────────────────────
def repl_mode():
    import time

    sys.stderr.write("[predict.py] Loading model artifacts...\n")
    t0 = time.perf_counter()
    model, scaler, feature_names = load_artifacts()
    dt = (time.perf_counter() - t0) * 1000
    sys.stderr.write(f"[predict.py] Artifacts loaded in {dt:.0f}ms. REPL ready.\n")

    # Signal readiness
    sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            mode = payload.get("mode", "predict")
            t0 = time.perf_counter()

            if mode == "predict":
                heat   = payload.get("heat", SAMPLE_HEAT)
                result = predict_composition(heat, model, scaler, feature_names)
                resp   = {"ok": True, "composition": result}

            elif mode == "batch_predict":
                heats  = payload.get("heats", [SAMPLE_HEAT])
                preds  = predict_batch(heats, model, scaler, feature_names)
                compositions = []
                for row in preds:
                    comp = {col: round(float(row[i]), 5) for i, col in enumerate(TARGET_COLS)}
                    compositions.append(comp)
                resp = {"ok": True, "compositions": compositions}

            elif mode == "optimise":
                heat    = payload.get("heat", SAMPLE_HEAT)
                targets = payload.get("targets", SAMPLE_TARGET)
                step    = int(payload.get("step", 50))
                result  = optimise_fa_hierarchical(heat, targets, model, scaler, feature_names, step=step, coarse_step=100)
                resp    = {"ok": True, **result}

            else:
                resp = {"ok": False, "error": f"Unknown mode: {mode}"}

            dt = (time.perf_counter() - t0) * 1000
            resp["time_ms"] = round(dt, 1)

        except Exception as e:
            resp = {"ok": False, "error": str(e)}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


# ─────────────────────────────────────────────
# LEGACY NODE MODE (stdin-once, kept for compat)
# ─────────────────────────────────────────────
def node_mode():
    payload = json.loads(sys.stdin.read())
    mode    = payload.get("mode", "predict")
    model, scaler, feature_names = load_artifacts()

    if mode == "predict":
        heat   = payload.get("heat", SAMPLE_HEAT)
        result = predict_composition(heat, model, scaler, feature_names)
        print(json.dumps({"ok": True, "composition": result}))

    elif mode == "optimise":
        heat         = payload.get("heat", SAMPLE_HEAT)
        targets      = payload.get("targets", SAMPLE_TARGET)
        step         = int(payload.get("step", 50))
        result       = optimise_fa_hierarchical(heat, targets, model, scaler, feature_names, step)
        print(json.dumps({"ok": True, **result}))


# ─────────────────────────────────────────────
# CLI MODE
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FerroMind Predict / Optimise")
    parser.add_argument("--mode",   choices=["predict", "optimise", "repl"], default="predict")
    parser.add_argument("--input",  type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--step",   type=int, default=50)
    args = parser.parse_args()

    if args.mode == "repl":
        repl_mode()
        return

    print("=" * 60)
    print("  FerroMind — Predict / Optimise")
    print("=" * 60)

    print("\nLoading model artifacts...")
    model, scaler, feature_names = load_artifacts()

    heat_dict = json.load(open(args.input)) if args.input else SAMPLE_HEAT
    if not args.input:
        print("  No --input file. Using built-in sample heat.")

    if args.mode == "predict":
        print("\nPredicting composition...")
        result = predict_composition(heat_dict, model, scaler, feature_names)
        print(f"\n  {'Element':<10} {'Predicted (wt%)':>16}")
        print(f"  {'-'*28}")
        for el, val in result.items():
            print(f"  {el:<10} {val:>16.5f}")

    elif args.mode == "optimise":
        target_range = json.load(open(args.target)) if args.target else SAMPLE_TARGET
        if not args.target:
            print("  No --target file. Using built-in sample target range.")
        print("\nRunning FA optimisation (vectorized + hierarchical)...")
        result = optimise_fa_hierarchical(heat_dict, target_range, model, scaler, feature_names, step=args.step)

        print(f"\n  Feasible      : {result['feasible']}")
        print(f"  FEMN Into Ladle : {result['FEMN Into Ladle']} kg")
        print(f"  SIMN Into Ladle : {result['SIMN Into Ladle']} kg")
        print(f"  FESI Into Ladle : {result['FESI Into Ladle']} kg")
        print(f"  Total FA        : {result['Total_FA_Ladle']} kg")
        print(f"\n  Predicted Composition vs Target:")
        for el, val in result["predicted"].items():
            bounds = target_range.get(el, ["-", "-"])
            if bounds[0] != "-":
                status = "✓" if bounds[0] <= val <= bounds[1] else "✗"
                print(f"    {status} {el:<6} {val:.5f}  (target: {bounds[0]}–{bounds[1]})")
            else:
                print(f"      {el:<6} {val:.5f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and "--mode" in sys.argv and "repl" in sys.argv:
        main()
    elif not sys.stdin.isatty():
        # Check if it's a single JSON payload or REPL
        # Default to node_mode for backward compat
        node_mode()
    else:
        main()