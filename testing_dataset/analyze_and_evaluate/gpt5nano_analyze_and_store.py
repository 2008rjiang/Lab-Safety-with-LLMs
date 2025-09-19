# analyze_and_store_gpt4o.py
# Purpose: Single-file script that:
#   - Reads local images from a folder (default: downloaded_images) with names like image0001.jpg
#   - Calls GPT-4o on each image
#   - Forces a strict JSON output format (Structured Outputs)
#   - Saves a consistent responses file to JSONL and CSV
#
# Usage:
#   1) pip install openai pandas tqdm pillow
#   2) python gpt5nano_analyze_and_store.py --images-dir test_images --out test_outputs/gpt5nano
#
# FOLDER REQUIRED:
#   downloaded_images/
#     image0001.jpg
#     image0002.png
#
# Notes:
# - Images are LOCAL files now (no CSV).
# - Paste your OpenAI API key below.

import os
import re
import csv
import json
import base64
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =========================
# PASTE YOUR API KEY HERE
# =========================
OPENAI_API_KEY = ...

# -------------------------
# Model to use
# -------------------------
MODEL  = "gpt-5-nano-2025-08-07"

# -------------------------
# Strict JSON schema (output blueprint)
# -------------------------
SCHEMA = {
    "name": "lab_assessment",
    "schema": {
        "type": "object",
        "properties": {
            "lab_type":   {"type": "string", "enum": ["bio","chem","ee"]},
            "is_safe":    {"type": "boolean"},
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": ["PPE","SOP","WO"]}
            },
            "reasoning":  {"type": "string"}
        },
        "required": ["lab_type","is_safe","categories","reasoning"],
        "additionalProperties": False
    },
    "strict": True
}


# -------------------------
# Prompt — must choose one of the lab types
# -------------------------
ANALYSIS_SYSTEM = "You are a lab-safety expert. Be concise, evidence-based."
ANALYSIS_USER = (
    "Analyze the provided lab image and return ONLY a JSON object with:\n"
    "- lab_type: choose the single closest match from \"bio\", \"chem\", or \"ee\".\n"
    "- is_safe: boolean overall verdict (true/false)\n"
    "- categories: array with any of [\"PPE\", \"SOP\", \"WO\"] visibly relevant\n"
    "- reasoning: 1-4 sentences citing concrete, visible cues (no boilerplate)\n"
    "Definitions:\n"
    "- PPE: personal protective equipment\n"
    "- SOP: procedural practices (chemical/biological/electrical handling)\n"
    "- WO: workspace organization/housekeeping\n"
    "You must choose one lab_type from the allowed list, even if uncertain.\n"
    "Output: Return only the JSON (no extra keys, no extra text)."
)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
ID_PATTERN = re.compile(r"^image\d+$")

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"): return "image/jpeg"
    if ext == ".png":            return "image/png"
    if ext == ".webp":           return "image/webp"
    if ext == ".bmp":            return "image/bmp"
    if ext == ".gif":            return "image/gif"
    # Fallback
    return "application/octet-stream"

def file_to_data_url(path: Path) -> str:
    mime = guess_mime(path)
    data = path.read_bytes()
    b64  = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

def call_model_with_local_image(client: OpenAI, model: str, img_path: Path) -> Tuple[dict, str]:
    """
    Calls the specified model with a local image by embedding it as a data URL.
    Returns (parsed_dict, raw_text).
    """
    data_url = file_to_data_url(img_path)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYSIS_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_USER},
                    {"type": "image_url", "image_url": {"url": data_url}},  # <-- changed here
                ],
            },
        ],
        response_format={"type": "json_schema", "json_schema": SCHEMA},
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"_raw": raw}
    return parsed, raw

def find_images(images_dir: Path):
    """
    Yield (image_id, image_path) for files under images_dir whose basename (without extension)
    matches ^image\\d+$ and extension is in VALID_EXTS.
    """
    for p in sorted(images_dir.iterdir()):
        if not p.is_file(): 
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        stem = p.stem
        if not ID_PATTERN.match(stem):
            continue
        yield stem, p

def main():
    # -------- args --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="downloaded_images", help="Folder with images named like image0001.jpg")
    parser.add_argument("--out", default="outputs", help="Output folder")
    args = parser.parse_args()

    # -------- client --------
    assert OPENAI_API_KEY and OPENAI_API_KEY != "api key", "Please paste a valid OpenAI API key in OPENAI_API_KEY."
    client = OpenAI(api_key=OPENAI_API_KEY)

    images_dir = Path(args.images_dir)
    assert images_dir.exists() and images_dir.is_dir(), f"Images folder not found: {images_dir}"

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # -------- gather images --------
    pairs = list(find_images(images_dir))
    assert pairs, f"No valid images found in {images_dir}. Expected files like image0001.jpg (jpg/png/webp/bmp/gif)."

    # -------- analyze --------
    rows = []
    for image_id, img_path in tqdm(pairs, total=len(pairs)):
        parsed, raw = call_model_with_local_image(client, MODEL, img_path)

        lab_type  = parsed.get("lab_type")
        is_safe   = parsed.get("is_safe")
        cats_list = parsed.get("categories", [])
        reasoning = parsed.get("reasoning", "")

        categories_csv = ",".join(cats_list) if isinstance(cats_list, list) else ""

        rows.append({
            "image_id": image_id,
            "file_path": str(img_path),
            "model": MODEL,
            "lab_type": lab_type,
            "is_safe": is_safe,
            "categories_csv": categories_csv,
            "reasoning": reasoning,
            "response_json": raw
        })

    # -------- write outputs --------
    jsonl_path = out_dir / "gpt5nano_model_responses.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = out_dir / "gpt5nano_model_responses.csv"
    pd.DataFrame(rows, columns=[
        "image_id","file_path","model","lab_type","is_safe","categories_csv","reasoning","response_json"
    ]).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote:\n - {jsonl_path}\n - {csv_path}\nAnalyzed {len(rows)} images from {images_dir}")

if __name__ == "__main__":
    main()
