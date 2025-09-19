#!/usr/bin/env python3
"""
Extract the top-K images matching a text prompt using CLIP similarity, print and save them,
then remove those entries from the embeddings NPZ to produce a new NPZ of remaining images.

Usage:
python extract_and_remove_topk.py --embeds remaining_embeddings.npz --prompt "students in a laboratory working on experiments" --k 15 --output_extract top50.txt --output_npz remaining_embeddings.npz

Requirements:
    pip install torch clip numpy
"""
import argparse
import numpy as np
import torch
import clip


def main():
    parser = argparse.ArgumentParser(
        description="Extract top-K similar images by prompt and remove them from NPZ"
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
        help='Number of top matches to extract'
    )
    parser.add_argument(
        '--output_extract', '-x', default='topk_urls.txt',
        help='Path to write extracted URLs and scores'
    )
    parser.add_argument(
        '--output_npz', '-o', default='remaining_embeddings.npz',
        help='Path for the NPZ file of remaining embeddings'
    )
    args = parser.parse_args()

    # Load embeddings and URLs
    data = np.load(args.embeds, allow_pickle=True)
    embeds = data['embeddings']  # shape (N,512)
    urls   = data['urls']        # shape (N,)
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
    topk_idxs = np.argsort(-sims)[:args.k]

    # Print and save top-K URLs with scores
    with open(args.output_extract, 'w') as f:
        for rank, idx in enumerate(topk_idxs, start=1):
            url = urls[idx]
            score = sims[idx]
            line = f"{rank:2d}. {url}   (score={score:.3f})"
            print(line)
            f.write(line + "\n")

    # Remove extracted indices from arrays
    mask = np.ones(len(urls), dtype=bool)
    mask[topk_idxs] = False
    remaining_embeds = embeds[mask]
    remaining_urls   = urls[mask]
    
    # Filter failed list if present
    if failed is not None:
        extracted_urls = set(urls[idx] for idx in topk_idxs)
        remaining_failed = [item for item in failed if item[1] not in extracted_urls]
    
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

    print(f"\nExtracted {len(topk_idxs)} URLs to '{args.output_extract}'.")
    print(f"Saved {len(remaining_urls)} remaining embeddings to '{args.output_npz}'.")


if __name__ == '__main__':
    main()
