# $env:OPENAI_API_KEY="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"

# python evaluate_reasoning.py --csv test_split.csv --jsonl test_outputs/gpt4o/gpt4o_model_responses.jsonl --out evaluation_outputs/gpt4o/gpt4o_reasoning_scores.csv

# python evaluate_reasoning.py --csv test_split.csv --jsonl test_outputs/gpt5nano/gpt5nano_model_responses.jsonl --out evaluation_outputs/gpt5nano/gpt5nano_reasoning_scores.csv

# python evaluate_reasoning.py --csv test_split.csv --jsonl test_outputs/gpt5mini/gpt5mini_model_responses.jsonl --out evaluation_outputs/gpt5mini/gpt5mini_reasoning_scores.csv

# python evaluate_reasoning.py --csv test_split.csv --jsonl test_outputs/gpto4mini/gpto4mini_model_responses.jsonl --out evaluation_outputs/gpto4mini/gpto4mini_reasoning_scores.csv

# python evaluate_reasoning.py --csv test_split.csv --jsonl test_outputs/qwen25vl/qwen25vl_model_responses.jsonl --out evaluation_outputs/qwen25vl/qwen25vl_reasoning_scores.csv



# evaluate_reasoning.py
# Purpose: Score model reasoning against ground-truth rationale on a 0..5 scale
#          AND explain the score choice in the output CSV.
#
# Output CSV columns:
#   image_id, reasoning_points, rubric_level, score_explanation
#
# Notes:
# - Reads GT rationales from --csv (column preference via --reason-col, fallback to gpt_reason/details).
# - Reads predicted reasoning from --jsonl (handles nested shapes).


import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================#
# Utilities & loading  #
# =====================#

IMG_EXT_RE = re.compile(r"\.(jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)
JSON_EXTRACT_RE = re.compile(r"\{[\s\S]*\}", re.M)

def normalize_image_id(x: str) -> str:
    s = str(x).strip().replace("\\", "/").split("/")[-1]
    return IMG_EXT_RE.sub("", s)

def load_ground_truth(csv_paths: List[str], reason_col_preference: str) -> pd.DataFrame:
    """
    Load only the ground-truth rationale (not written to output).

    Note:
    - In your example.csv, only **unsafe** images have a "details" (or rationale) column
      describing what makes them unsafe.
    - **Safe** images are marked as safe and therefore typically do **not** contain a
      "details" entry; this absence is expected and should not be treated as missing data.
    """
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "image_id" not in df.columns:
            raise ValueError(f"{p}: missing 'image_id'")

        df["image_id"] = df["image_id"].apply(normalize_image_id)

        ren = {}
        for col in df.columns:
            lc = col.strip().lower()
            if lc == "gpt reason":
                ren[col] = "gpt_reason"
        if ren:
            df = df.rename(columns=ren)

        # Prefer reason_col_preference if present, else fallback
        if reason_col_preference in df.columns:
            use_col = reason_col_preference
        elif "gpt_reason" in df.columns:
            use_col = "gpt_reason"
        elif "details" in df.columns:
            use_col = "details"
        else:
            # For safe images, no details are expected → fill with empty
            df["details"] = ""
            use_col = "details"

        frames.append(df[["image_id", use_col]].rename(columns={use_col: "gt_rationale"}))

    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["image_id"], keep="last").reset_index(drop=True)

def load_model_reasoning(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue

            image_id = obj.get("image_id")
            if not image_id:
                continue

            reasoning = None
            if isinstance(obj.get("lab_assessment"), dict):
                reasoning = obj["lab_assessment"].get("reasoning")
            elif isinstance(obj.get("parsed"), dict):
                la = obj["parsed"].get("lab_assessment", obj["parsed"])
                if isinstance(la, dict):
                    reasoning = la.get("reasoning")
            if reasoning is None and "reasoning" in obj:
                reasoning = obj.get("reasoning")
            if reasoning is None and isinstance(obj.get("response_json"), str) and obj["response_json"].strip():
                try:
                    parsed = json.loads(obj["response_json"])
                    if isinstance(parsed, dict):
                        reasoning = parsed.get("reasoning", reasoning)
                except Exception:
                    pass

            rows.append({
                "image_id": normalize_image_id(image_id),
                "pred_reasoning": "" if reasoning is None else str(reasoning),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable predictions parsed from {jsonl_path}.")
    return df.drop_duplicates(subset=["image_id"], keep="last").reset_index(drop=True)

def load_optional_context(context_paths: List[str]) -> dict:
    ctx = {}
    for p in context_paths or []:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "image_id" not in df.columns:
            continue
        drop_cols = [c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.len().mean() > 200]
        slim = df.drop(columns=drop_cols, errors="ignore").copy()
        for _, row in slim.iterrows():
            iid = normalize_image_id(row["image_id"])
            small = {k: row[k] for k in slim.columns if k != "image_id"}
            for k, v in list(small.items()):
                if isinstance(v, float) and np.isnan(v):
                    small[k] = None
                else:
                    small[k] = None if v is None else str(v)
            ctx.setdefault(iid, {}).update(small)
    return ctx

# =====================#
# Scoring prompt/rubric#
# =====================#

SCORER_SYSTEM = (
    "You are a strict evaluator for lab-image reasoning. "
    "You will receive a ground-truth rationale (human-written) and a model's predicted reasoning. "
    "Score how well the predicted reasoning matches the ground-truth rationale. "
    "Focus on correctness, specificity, and whether the reasoning justifies the labels "
    "(safety status, lab type, categories PPE/SOP/WO). "
    "Return ONLY a JSON object with keys exactly: "
    "\"reasoning_points\" (0..5 number), \"rubric_level\" (string), \"explanation\" (string). "
    "Important dataset note: in these CSVs, only UNSAFE images typically include a 'details' rationale. "
    "SAFE images usually have no details text. This absence is expected and should NOT be penalized."
)

RUBRIC_BLOCK = """
Assign REASONING points on a 0..5 scale. Pick the single best level and justify briefly.

Levels:

5 — Exceptional
- Fully aligns with GT: correct safety status, lab type, and categories.
- Cites all key GT drivers with concrete evidence.
- No contradictions/hallucinations.
Examples: 
  • GT unsafe (Chem): “missing goggles, missing gloves (PPE)”; model notes absent eye/hand protection.  
  • GT unsafe (Bio): “improper footwear, loose hair”; model points out heels + loose hair.  
  • GT safe (EE): model notes glasses, controlled wiring, organized bench with no invented hazards.

4 — Strong
- Correct on major drivers and labels; may miss ONE minor GT detail.
- No contradictions/hallucinations.
Examples: notes goggles but not gloves; or heels but not loose hair.

3 — Adequate
- Gets the main idea (safe/unsafe correct) and at least ONE specific detail.
- Misses multiple GT details or fairly generic.
Example: “Missing PPE makes it unsafe” (unspecified) when GT called out goggles + gloves.

2 — Partial
- Some correct elements, but incomplete/vague OR includes minor inaccuracies.
- Weak linkage to GT labels.
Example: says “safety issue” but misattributes (clutter vs missing goggles).

1 — Weak
- Vague/shallow; little GT evidence; could mislead labels.
Example: “Seems unsafe” with no supporting detail.

0 — Incorrect/Misleading
- Contradicts GT or hallucinates hazards/safety.
Examples: GT unsafe → model says “safe with goggles”; GT safe → model invents chemical spills.
"""

USER_TEMPLATE = """
GROUND_TRUTH_RATIONALE:
{gt}

PREDICTED_REASONING:
{pred}

OPTIONAL CONTEXT (for understanding only):
{ctx}

TASK:
Score ONLY how well PREDICTED_REASONING aligns with GROUND_TRUTH_RATIONALE and implied labels (safety status, lab type, PPE/SOP/WO).
Return JSON ONLY with:
- "reasoning_points" (0..5 integer),
- "rubric_level" (Exceptional, Strong, Adequate, Partial, Weak, Incorrect/Misleading),
- "explanation" (short, cite specific overlaps/gaps).
"""


# =====================#
# OpenAI integration   #
# =====================#

def get_client(api_key: Optional[str]):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def extract_json_maybe(s: str) -> Optional[dict]:
    if not s:
        return None
    m = JSON_EXTRACT_RE.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def clamp_0_5(x) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    if np.isnan(v):
        return 0
    v = max(0.0, min(5.0, v))
    return int(round(v))

def call_gpt5_points_and_expl(
    client,
    gt_text: str,
    pred_text: str,
    context_str: str,
    retries: int = 2,
    debug_file=None
) -> Tuple[int, str, str]:
    """
    Returns: (points_0_5, rubric_level, explanation)
    Falls back to (0, "Weak", "No valid JSON returned") on failure.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": SCORER_SYSTEM},
                    {"role": "system", "content": RUBRIC_BLOCK},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        gt=(gt_text or "(none)"),
                        pred=(pred_text or "(none)"),
                        ctx=(context_str or "(none)")
                    )},
                ],
            )
            content = resp.choices[0].message.content
            data = extract_json_maybe(content)

            if debug_file is not None:
                debug_file.write(json.dumps({
                    "attempt": attempt,
                    "raw_response": content
                }, ensure_ascii=False) + "\n")

            if not data:
                last_err = ValueError("no JSON")
                time.sleep(0.25)
                continue

            pts = clamp_0_5(data.get("reasoning_points"))
            level = str(data.get("rubric_level") or "").strip()
            expl = str(data.get("explanation") or "").strip()

            # basic sanitation
            allowed = {"Exceptional","Strong","Adequate","Partial","Weak","Incorrect/Misleading"}
            if level not in allowed:
                # map roughly by score
                level = (
                    "Exceptional" if pts == 5 else
                    "Strong" if pts == 4 else
                    "Adequate" if pts == 3 else
                    "Partial" if pts == 2 else
                    "Weak" if pts == 1 else
                    "Incorrect/Misleading"
                )
            if not expl:
                expl = "No explanation provided by scorer."

            return pts, level, expl

        except Exception as e:
            last_err = e
            time.sleep(0.4)

    if debug_file is not None:
        dbg = {"error": str(last_err)}
        debug_file.write(json.dumps(dbg, ensure_ascii=False) + "\n")

    return 0, "Weak", "No valid JSON returned."

# =====================#
# Main                 #
# =====================#

def main():
    ap = argparse.ArgumentParser(
    description="Reasoning-only scoring (0..5 with explanation). "
                "Note: In ground-truth CSVs, only unsafe images have a 'details' rationale. "
                "Safe images usually have no details column, and this absence is expected."
)

    ap.add_argument("--csv", nargs="+", required=True, help="Ground-truth CSV(s) containing 'image_id' and rationale column.")
    ap.add_argument("--jsonl", required=True, help="Model JSONL containing predicted reasoning.")
    ap.add_argument("--reason-col", default="details", help="GT rationale column to use (default: details). Fallback to gpt_reason, details.")
    ap.add_argument("--out", default="evaluation_outputs/reasoning_scores.csv", help="Output CSV path.")
    ap.add_argument("--api-key", default=None, help="OpenAI API key (else env var OPENAI_API_KEY).")
    ap.add_argument("--context", nargs="*", default=None, help="Optional CSVs (e.g., core label results) passed as compact context.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between rows.")
    ap.add_argument("--max", type=int, default=None, help="Only score first N rows.")
    ap.add_argument("--debug-log", default="evaluation_outputs/reasoning_debug.jsonl", help="Debug JSONL of raw model outputs.")
    args = ap.parse_args()

    df_gt = load_ground_truth(args.csv, args.reason_col)
    df_pred = load_model_reasoning(args.jsonl)
    df = df_gt.merge(df_pred, on="image_id", how="inner")

    ctx_map = load_optional_context(args.context) if args.context else {}

    if args.max is not None:
        df = df.head(args.max).copy()

    client = get_client(args.api_key)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.debug_log).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.debug_log, "a", encoding="utf-8") as dbg:
        print("Scoring reasoning (0..5) with explanations...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            gt = row.get("gt_rationale") or ""
            pred = row.get("pred_reasoning") or ""
            iid = row["image_id"]
            context_str = json.dumps(ctx_map.get(iid, {}), ensure_ascii=False)

            pts, level, expl = call_gpt5_points_and_expl(
                client, gt, pred, context_str, retries=2, debug_file=dbg
            )
            rows.append({
                "image_id": iid,
                "reasoning_points": pts,         # 0..5
                "rubric_level": level,           # text level
                "score_explanation": expl,       # why the score
            })
            if args.sleep > 0:
                time.sleep(args.sleep)

    out = pd.DataFrame(rows, columns=["image_id","reasoning_points","rubric_level","score_explanation"])
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {args.out}")
    print("Each row includes a 0..5 score, rubric level, and short explanation.")
    print(f"Debug responses saved to: {args.debug_log}")

if __name__ == "__main__":
    main()
