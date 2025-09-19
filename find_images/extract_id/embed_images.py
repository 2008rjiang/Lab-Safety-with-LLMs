#!/usr/bin/env python3
"""
Embed images from a URL list using CLIP (float32 precision) with batched inference.

This version also logs all failed URLs and reasons to a separate .txt file.

Requirements:
    pip install torch torchvision ftfy regex tqdm requests pillow git+https://github.com/openai/CLIP.git

Usage:
python embed_images.py --input valid9.txt --output valid9.npz --batch_size 64
"""

import argparse
import os
import requests
from PIL import Image
from io import BytesIO
import torch
import clip
import numpy as np
from tqdm import tqdm


def load_urls(input_path):
    """Load URLs from a .txt or .csv file."""
    urls = []
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.txt':
        with open(input_path, 'r') as f:
            for line in f:
                url = line.strip()
                if url:
                    urls.append(url)
    elif ext == '.csv':
        import csv
        with open(input_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            headers = [h.lower() for h in header]
            url_idx = headers.index('url') if 'url' in headers else 0
            for row in reader:
                url = row[url_idx].strip()
                if url:
                    urls.append(url)
    else:
        raise ValueError(f"Unsupported input file type: {ext}")
    return urls


def download_image(url, timeout=10):
    """Fetch an image from URL and convert to RGB PIL.Image, returning img or error."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '')
        if 'image' not in content_type:
            raise ValueError(f"Not an image: {content_type}")
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        return img, None
    except Exception as e:
        return None, str(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed images using CLIP with batching, logging failures.")
    parser.add_argument('--input', '-i', required=True, help='Path to txt or csv file with image URLs')
    parser.add_argument('--output', '-o', default='embeddings.npz', help='Output .npz file')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model, preprocess = clip.load("ViT-B/32", device=device)

    urls = load_urls(args.input)
    N = len(urls)
    print(f"Found {N} URLs")

    embeddings = np.zeros((N, 512), dtype=np.float32)
    failed = []  # list of (index, url, error)

    batch_imgs = []
    batch_idxs = []
    for idx, url in enumerate(tqdm(urls, desc="Downloading & embedding")):
        img, err = download_image(url)
        if img is None:
            print(f"[ERROR] {url} → {err}")
            failed.append((idx, url, err))
        else:
            batch_imgs.append(preprocess(img))
            batch_idxs.append(idx)

        if len(batch_imgs) >= args.batch_size:
            batch_tensor = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                embs = model.encode_image(batch_tensor).cpu().numpy().astype(np.float32)
            for j, e in zip(batch_idxs, embs):
                embeddings[j] = e
            batch_imgs.clear()
            batch_idxs.clear()

    # flush remaining
    if batch_imgs:
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch_tensor).cpu().numpy().astype(np.float32)
        for j, e in zip(batch_idxs, embs):
            embeddings[j] = e

    # save NPZ results
    np.savez_compressed(args.output,
                        embeddings=embeddings,
                        urls=np.array(urls, dtype=object),
                        failed=np.array(failed, dtype=object))
    print(f"Saved embeddings to {args.output}")

    # write failures to a text file
    if failed:
        base = os.path.splitext(args.input)[0]
        failed_txt = f"{base}_failed.txt"
        with open(failed_txt, 'w') as f_out:
            for idx, url, err in failed:
                f_out.write(f"{idx}\t{url}\t{err}\n")
        print(f"Wrote {len(failed)} failures to {failed_txt}")
    else:
        print("No failures encountered.")
