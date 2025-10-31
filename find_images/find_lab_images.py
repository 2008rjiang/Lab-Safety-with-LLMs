'''
find_lab_images.py

Optimized scan of LAION-400M Parquet using PyArrow dataset for chunked reads,
arrow-level substring filtering via compute.match_substring, and CLIP embedding.
'''
import os
import csv
import torch
import clip
from tqdm import tqdm
import pyarrow.dataset as ds
import pyarrow.compute as pc

# ─── CONFIG ────────────────────────────────────────────────────────────────────
LAION_DIR    = r"D:\laion\test"        # Folder with your Parquet files
LAB_KEYWORDS = [                          # Keywords to filter captions
    "lab", "laboratory"
    # "chemistry", "biology", "engineering",
    # "beaker", "pipette", "microscope", "test tube", "fume hood",
    # "bunsen burner", "gloves", "safety goggles"
]
QUERY_TEXT   = "students working in a lab"
TOP_K        = 150
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
ROW_BATCH    = 10000    # much larger
CAP_BATCH    = 512
CSV_OUT      = "top_lab_images.csv"
EARLY_STOP   = 0.95     # stop if 10th best sim ≥ this
# ────────────────────────────────────────────────────────────────────────────────

# load CLIP 
device = torch.device(DEVICE)
model, _ = clip.load("ViT-B/32", device)
model.eval()
with torch.no_grad():
    q_tok  = clip.tokenize([QUERY_TEXT], truncate=True).to(device)
    q_emb  = model.encode_text(q_tok)
    q_emb /= q_emb.norm(dim=-1, keepdim=True)

# build dataset + pushdown filter
dataset = ds.dataset(LAION_DIR, format="parquet")
predicate = None
for kw in LAB_KEYWORDS:
    # ✅ use the compute function, passing an Expression
    expr = pc.match_substring(pc.field("TEXT"), pattern=kw, ignore_case=True)
    predicate = expr if predicate is None else (predicate | expr)

reader = dataset.to_batches(
    filter=predicate,
    columns=["URL","TEXT"],
    batch_size=ROW_BATCH
)

top_matches = []
for batch_idx, batch in enumerate(tqdm(reader, desc="Scanning")):
    urls = batch.column("URL").to_pylist()
    caps = batch.column("TEXT").to_pylist()

    # embed in sub-batches
    for i in range(0, len(caps), CAP_BATCH):
        sub_urls = urls[i:i+CAP_BATCH]
        sub_caps = caps[i:i+CAP_BATCH]
        with torch.no_grad():
            toks = clip.tokenize(sub_caps, truncate=True).to(device)
            emb  = model.encode_text(toks)
            emb /= emb.norm(dim=-1, keepdim=True)
            sims = (emb @ q_emb.T).squeeze(1).cpu().numpy()

        for sim, u, c in zip(sims, sub_urls, sub_caps):
            top_matches.append((float(sim), u, c))

    # keep only the best TOP_K so far
    top_matches = sorted(top_matches, key=lambda x: x[0], reverse=True)[:TOP_K]

    # print a live threshold every 50 batches
    if batch_idx % 50 == 0:
        print(f"[batch {batch_idx}] 10th best sim = {top_matches[-1][0]:.4f}")

    # early stop
    if top_matches[-1][0] >= EARLY_STOP:
        print(f"Reached threshold {EARLY_STOP:.2f} → stopping early.")
        break

# write out results    
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["score","url","caption"])
    for score, url, cap in top_matches:
        writer.writerow([f"{score:.6f}", url, cap])

print(f"Saved top {TOP_K} matches to {CSV_OUT}")