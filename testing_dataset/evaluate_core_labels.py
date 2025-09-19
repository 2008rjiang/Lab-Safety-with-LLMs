# 4o
# python evaluate_core_labels.py --csv test_split.csv --jsonl test_outputs/gpt4o/gpt4o_model_responses.jsonl --out evaluation_outputs/gpt4o/gpt4o_core_label_comparison.csv
# Items: 300
# Safety accuracy: 0.663
# Lab type accuracy: 0.850
# Category exact match: 0.644
# Category macro F1: 0.740
 
# 5 nano
# python evaluate_core_labels.py --csv test_split.csv --jsonl test_outputs/gpt5nano/gpt5nano_model_responses.jsonl --out evaluation_outputs/gpt5nano/gpt5nano_core_label_comparison.csv
# Items: 300
# Safety accuracy: 0.627
# Lab type accuracy: 0.803
# Category exact match: 0.624
# Category macro F1: 0.725

# 5 mini
# python evaluate_core_labels.py --csv test_split.csv --jsonl test_outputs/gpt5mini/gpt5mini_model_responses.jsonl --out evaluation_outputs/gpt5mini/gpt5mini_core_label_comparison.csv
# Items: 300
# Safety accuracy: 0.637
# Lab type accuracy: 0.840
# Category exact match: 0.642
# Category macro F1: 0.748

# o4 mini
# python evaluate_core_labels.py --csv test_split.csv --jsonl test_outputs/gpto4mini/gpto4mini_model_responses.jsonl --out evaluation_outputs/gpto4mini/gpto4mini_core_label_comparison.csv
# Items: 300
# Safety accuracy: 0.697
# Lab type accuracy: 0.830
# Category exact match: 0.603
# Category macro F1: 0.703

# qwen
# python evaluate_core_labels.py --csv test_split.csv --jsonl test_outputs/qwen25vl/qwen25vl_model_responses.jsonl --out evaluation_outputs/qwen25vl/qwen25vl_core_label_comparison.csv
# Items: 300
# Safety accuracy: 0.503
# Lab type accuracy: 0.737
# Category exact match: 0.449
# Category macro F1: 0.550


import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================  
# Config (strict categories)
# =========================
CAT_ALLOWED = {"PPE", "SOP", "WO"}  # case-insensitive input -> uppercase
LAB_ALLOWED_PRED = {"bio", "chem", "ee"}  # predictions expected as these
LAB_GT_MAP = {  # CSV lab type: biology/chemistry/ee -> bio/chem/ee
    "biology": "bio",
    "chemistry": "chem",
    "ee": "ee",
}

IMG_EXT_RE = re.compile(r"\.(jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)

# ===== NEW: weights =====
W_SAFETY = 2
W_LAB    = 1
W_CAT    = 1
MAX_POINTS = W_SAFETY + W_LAB + W_CAT  # = 4

# =========================
# Helpers
# =========================
def normalize_image_id(x: str) -> str:
    """
    Make 'image0007.jpg' or 'path/to/image0007.png' -> 'image0007'
    """
    s = str(x).strip().replace("\\", "/").split("/")[-1]
    s = IMG_EXT_RE.sub("", s)
    return s


def norm_lab_gt(s: str) -> Optional[str]:
    if s is None:
        return None
    key = re.sub(r"[^a-z]", "", str(s).strip().lower())  # "Biology" -> "biology"
    return LAB_GT_MAP.get(key)


def norm_lab_pred(s: str) -> Optional[str]:
    if s is None:
        return None
    key = re.sub(r"[^a-z]", "", str(s).strip().lower())
    return key if key in LAB_ALLOWED_PRED else None


def norm_is_safe_from_unsafe(x) -> Optional[bool]:
    """
    CSV 'unsafe': 1 or 1.0 => is_safe=False; 0 or 0.0 => is_safe=True
    Handles strings like '0.0', '1', floats, ints.
    """
    try:
        if x is None:
            return None
        xi = int(float(x))
        return False if xi == 1 else True
    except Exception:
        # also handle pandas NA
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return None


def split_categories_strict(raw) -> List[str]:
    """
    Accepts single or multiple categories.
    Keeps ONLY PPE/SOP/WO (case-insensitive). Dedupes preserving order.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    parts = re.split(r"[,\;/\|\s]+", s)  # comma/semicolon/slash/pipe/whitespace
    out, seen = [], set()
    for p in parts:
        token = p.strip().upper()
        if token in CAT_ALLOWED and token not in seen:
            out.append(token)
            seen.add(token)
    return out


def _seq_from_any(val):
    """
    Convert a merged cell to a sequence of tokens without raising 'ambiguous truth value' errors.
    - list/tuple/set -> as-is
    - numpy array -> tolist()
    - str -> split on delimiters
    - NaN/None -> empty list
    - anything else -> [val] if looks scalar (but we filter tokens anyway)
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        return re.split(r"[,\;/\|\s]+", val.strip())
    # scalar NaN?
    try:
        if pd.isna(val):
            return []
    except Exception:
        pass
    return [val]


def to_catset(val) -> Set[str]:
    """
    Robustly convert merged cell to a set of categories.
    Handles list, np.ndarray, str, NaN, etc.
    """
    seq = _seq_from_any(val)
    out = set()
    for p in seq:
        token = str(p).strip().upper()
        if token in CAT_ALLOWED:
            out.add(token)
    return out


def list_to_str(v) -> str:
    """
    For output: normalize list/ndarray/str/NaN into a ';'-joined string of PPE/SOP/WO.
    """
    seq = _seq_from_any(v)
    seen, out = set(), []
    for p in seq:
        token = str(p).strip().upper()
        if token in CAT_ALLOWED and token not in seen:
            seen.add(token)
            out.append(token)
    return ";".join(out)


def prf1(pred: Set[str], gt: Set[str]) -> Tuple[float, float, float]:
    if len(pred) == 0 and len(gt) == 0:
        return 1.0, 1.0, 1.0
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# =========================
# Loaders
# =========================
def load_ground_truth(csv_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "image_id" not in df.columns:
            raise ValueError(f"{p}: missing 'image_id' column")

        # standardize two column names if present
        ren = {}
        for col in df.columns:
            lc = col.strip().lower()
            if lc == "lab type":
                ren[col] = "lab_type"
            if lc == "gpt reason":
                ren[col] = "gpt_reason"
        if ren:
            df = df.rename(columns=ren)

        # Normalize IDs (strip extensions)
        df["image_id"] = df["image_id"].apply(normalize_image_id)

        # Canonicalize labels
        df["lab_type_gt"] = df.get("lab_type", "").apply(norm_lab_gt)
        df["is_safe_gt"]  = df.get("unsafe").apply(norm_is_safe_from_unsafe)
        df["categories_gt"] = df.get("category", "").apply(split_categories_strict)

        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    # light sanity warnings
    if full["lab_type_gt"].isna().sum() > 0:
        print("[WARN] Some rows have lab_type not in {biology, chemistry, ee}.")
    if full["is_safe_gt"].isna().sum() > 0:
        print("[WARN] Some rows have invalid 'unsafe' values.")

    return full


def load_model_jsonl(jsonl_path: str) -> pd.DataFrame:
    """
    Supports:
      - flat fields at top
      - nested 'lab_assessment' or 'parsed.lab_assessment'
      - stringified JSON in 'response_json'
      - fallback to 'categories_csv' if only a single string exists
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)

            image_id = obj.get("image_id")
            if not image_id:
                continue

            la = None
            if isinstance(obj.get("lab_assessment"), dict):
                la = obj["lab_assessment"]
            elif isinstance(obj.get("parsed"), dict):
                la = obj["parsed"].get("lab_assessment", obj["parsed"])

            if la is None:
                keys = {"lab_type", "is_safe", "categories", "reasoning"}
                if keys.issubset(set(obj.keys())):
                    la = {k: obj[k] for k in keys}

            if la is None and isinstance(obj.get("response_json"), str) and obj["response_json"].strip():
                try:
                    parsed = json.loads(obj["response_json"])
                    if {"lab_type","is_safe","categories","reasoning"}.issubset(parsed.keys()):
                        la = parsed
                except Exception:
                    la = None

            if la is None:
                la = {}
                if "lab_type" in obj: la["lab_type"] = obj["lab_type"]
                if "is_safe" in obj: la["is_safe"] = obj["is_safe"]
                if "categories" in obj:
                    la["categories"] = obj["categories"]
                elif "categories_csv" in obj and obj["categories_csv"]:
                    la["categories"] = [obj["categories_csv"]]
                if not any(k in la for k in ("lab_type","is_safe","categories")):
                    continue

            lab_type = norm_lab_pred(la.get("lab_type"))
            is_safe = la.get("is_safe", None)
            if isinstance(is_safe, str):
                is_safe = is_safe.strip().lower() in {"true","1","yes"}
            elif not isinstance(is_safe, bool):
                is_safe = None

            cats_raw = la.get("categories", [])
            if isinstance(cats_raw, list):
                categories = split_categories_strict(",".join(map(str, cats_raw)))
            else:
                categories = split_categories_strict(str(cats_raw))

            rows.append({
                "image_id": normalize_image_id(image_id),
                "lab_type_pred": lab_type,
                "is_safe_pred": is_safe,
                "categories_pred": categories,
            })

    df = pd.DataFrame(rows)
    if df.empty or "image_id" not in df.columns:
        raise ValueError(f"No usable predictions parsed from {jsonl_path}.")
    df = df.drop_duplicates(subset=["image_id"], keep="last").reset_index(drop=True)
    return df


# =========================
# Evaluation (no reasoning)
# =========================
def evaluate_core(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    merged = df_gt.merge(df_pred, on="image_id", how="left", suffixes=("", "_predonly"))

    safety_points = []
    lab_points = []
    cat_points = []

    cat_precision = []
    cat_recall = []
    cat_f1 = []

    print("Scoring rows...")
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        # Safety (0 or 2 points)
        s_gt = row.get("is_safe_gt")
        s_pd = row.get("is_safe_pred")
        s_ok = (isinstance(s_gt, bool) and isinstance(s_pd, bool) and s_gt == s_pd)
        safety_points.append(2 if s_ok else 0)

        # Lab type (0 or 1 point)
        l_gt = row.get("lab_type_gt")
        l_pd = row.get("lab_type_pred")
        l_ok = (l_gt is not None and l_pd is not None and l_gt == l_pd)
        lab_points.append(1 if l_ok else 0)

        # Categories (multi-label, exact match; 0 or 1 point)
        gt = to_catset(row.get("categories_gt"))
        pd_ = to_catset(row.get("categories_pred"))
        p, r, f = prf1(pd_, gt)

        tp = len(pd_ & gt)
        fp = len(pd_ - gt)
        fn = len(gt - pd_)
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0

        cat_precision.append(p)
        cat_recall.append(r)
        cat_f1.append(f)

        # Replace exact match:
        # cat_points.append(1 if gt == pd_ else 0)
        # With proportional points:
        cat_points.append(jaccard)  # in [0,1], proportional to overlap


    out = pd.DataFrame({
        "image_id": merged["image_id"],
        "lab_type_gt": merged["lab_type_gt"],
        "lab_type_pred": merged["lab_type_pred"],
        "lab_points": lab_points,
        "is_safe_gt": merged["is_safe_gt"],
        "is_safe_pred": merged["is_safe_pred"],
        "safety_points": safety_points,
        "categories_gt": merged["categories_gt"].apply(list_to_str),
        "categories_pred": merged["categories_pred"].apply(list_to_str),
        "cat_precision": cat_precision,
        "cat_recall": cat_recall,
        "cat_f1": cat_f1,
        "cat_points": cat_points,
    })
    return out



# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Compare safety, lab type, and category between CSV ground truth and GPT-4o JSONL (strict enums).")
    ap.add_argument("--csv", nargs="+", required=True, help="CSV file(s) with columns: image_id,url,category,lab type,unsafe,...")
    ap.add_argument("--jsonl", default=str(Path("gpt4o_test_outputs") / "model_responses.jsonl"),
                    help="Path to GPT-4o JSONL (default: gpt4o_test_outputs/model_responses.jsonl).")
    ap.add_argument("--out", default="evaluation_outputs/core_label_comparison.csv",
                    help="Output CSV (default: evaluation_outputs/core_label_comparison.csv)")
    args = ap.parse_args()

    df_gt = load_ground_truth(args.csv)
    df_pred = load_model_jsonl(args.jsonl)

    if "image_id" not in df_pred.columns or df_pred.empty:
        raise ValueError("Predictions parsed empty or missing 'image_id'.")

    results = evaluate_core(df_gt, df_pred)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False, encoding="utf-8")

    # quick summary
    print("\n=== CORE LABEL SUMMARY ===")
    print(f"Items: {len(results)}")

    # Convert points back to accuracies (since points are 0/2, 0/1, 0/1)
    safety_accuracy = (results["safety_points"] / 2).mean() if len(results) else 0.0
    lab_type_accuracy = results["lab_points"].mean() if len(results) else 0.0
    cat_exact_match = results["cat_points"].mean() if len(results) else 0.0
    print(f"Safety accuracy: {safety_accuracy:.3f}")
    print(f"Lab type accuracy: {lab_type_accuracy:.3f}")
    print(f"Category exact match: {cat_exact_match:.3f}")
    print(f"Category macro F1: {results['cat_f1'].mean():.3f}")
    print(f"\nWrote: {args.out}")



if __name__ == "__main__":
    main()
