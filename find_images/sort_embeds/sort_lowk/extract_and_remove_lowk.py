#!/usr/bin/env python3
"""
Extract the top-K images matching a text prompt using CLIP similarity, print and save them,
separate out any embeddings with NaN scores into their own NPZ, then remove both extracted top-K
and NaN entries from the original NPZ to produce a new NPZ of remaining images.

Usage:
python extract_and_remove_lowk.py --embeds valid_all_good.npz --prompt "unsafe laboratory conditions with spilled chemicals and cluttered desks" --k 10000 --output_extract lowToHigh_prompt5.txt --output_npz remaining_embeddings.npz --output_nan_npz nan_embeddings_prompt5.npz

Requirements:
    pip install torch clip numpy


Prompts:
1. students in a laboratory working on experiments
2. students wearing lab coats, gloves, and goggles in a laboratory
3. college students conducting electrical engineering experiments in a lab
4. researchers performing experiments with chemicals and lab equipment
5. unsafe laboratory conditions with spilled chemicals and cluttered desks

"""

import argparse
import numpy as np
import torch
import clip

def main():
    parser = argparse.ArgumentParser(
        description="Extract bottom-K similar images by prompt, separate NaNs, and remove them"
    )
    parser.add_argument(
        '--embeds', '-e', required=True,
        help='Input NPZ file with arrays "embeddings", "urls", optional "failed"'
    )
    parser.add_argument(
        '--prompt', '-p', required=True,
        help='Text prompt to score against image embeddings'
    )
    parser.add_argument(
        '--k', '-k', type=int, default=50,
        help='Number of bottom matches to extract'
    )
    parser.add_argument(
        '--output_extract', '-x', default='bottomk_urls.txt',
        help='Path to write extracted URLs and scores'
    )
    parser.add_argument(
        '--output_npz', '-o', default='remaining_embeddings.npz',
        help='Path for the NPZ file of remaining embeddings'
    )
    parser.add_argument(
        '--output_nan_npz', '-n', default='nan_embeddings.npz',
        help='Path for the NPZ file of embeddings with NaN scores'
    )
    args = parser.parse_args()

    # Load embeddings and URLs
    data = np.load(args.embeds, allow_pickle=True)
    embeds = data['embeddings']    # shape (N,512)
    urls   = data['urls']          # shape (N,)
    failed = data['failed'] if 'failed' in data else None

    # Normalize image embeddings
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    img_norms = embeds / norms

    # Load CLIP and embed text prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    token = clip.tokenize([args.prompt]).to(device)
    with torch.no_grad():
        txt_emb = model.encode_text(token).cpu().numpy()[0]
    txt_norm = txt_emb / np.linalg.norm(txt_emb)

    # Compute cosine similarities
    sims = img_norms.dot(txt_norm)  # shape (N,)

    # Separate NaN-score entries
    nan_mask = np.isnan(sims)
    nan_idxs = np.where(nan_mask)[0]
    if len(nan_idxs) > 0:
        nan_embeds = embeds[nan_idxs]
        nan_urls   = urls[nan_idxs]
        if failed is not None:
            # filter failed entries not in extracted
            nan_failed = [item for item in failed if item[1] in set(nan_urls)]
        np.savez_compressed(
            args.output_nan_npz,
            embeddings=nan_embeds,
            urls=nan_urls,
            **({'failed': np.array(nan_failed, dtype=object)} if failed is not None else {})
        )
        print(f"Saved {len(nan_urls)} NaN-score entries to '{args.output_nan_npz}'.")
    else:
        print("No NaN scores found.")

    # Mask out NaNs for further processing
    valid_mask = ~nan_mask
    valid_idxs = np.where(valid_mask)[0]
    valid_sims = sims[valid_mask]

    # Extract bottom-K (lowest) scores
    k = min(args.k, len(valid_sims))
    bottom_rel = np.argsort(valid_sims)[:k]
    bottom_idxs = valid_idxs[bottom_rel]

    # Print and save bottom-K URLs with scores
    with open(args.output_extract, 'w') as f:
        for rank, idx in enumerate(bottom_idxs, start=1):
            url = urls[idx]
            score = sims[idx]
            line = f"{rank:2d}. {url}   (score={score:.3f})"
            print(line)
            f.write(line + "\n")

    # Build mask to remove both NaNs and extracted bottom-K
    remove_mask = np.zeros(len(urls), dtype=bool)
    remove_mask[nan_idxs] = True
    remove_mask[bottom_idxs] = True

    remaining_embeds = embeds[~remove_mask]
    remaining_urls   = urls[~remove_mask]

    # Filter failed list if present
    if failed is not None:
        removed_urls = set(urls[np.concatenate([nan_idxs, bottom_idxs])])
        remaining_failed = [item for item in failed if item[1] not in removed_urls]

    # Save remaining to new NPZ
    if failed is not None:
        np.savez_compressed(
            args.output_npz,
            embeddings=remaining_embeds,
            urls=remaining_urls,
            failed=np.array(remaining_failed, dtype=object)
        )
    else:
        np.savez_compressed(
            args.output_npz,
            embeddings=remaining_embeds,
            urls=remaining_urls
        )

    print(f"\nExtracted {k} lowest-score URLs to '{args.output_extract}'.")
    print(f"Saved {len(remaining_urls)} remaining embeddings to '{args.output_npz}'.")

if __name__ == '__main__':
    main()
