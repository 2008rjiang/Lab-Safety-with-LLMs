#!/usr/bin/env python3
"""
Filter and group laboratory image URLs by resolution quality, saving a CSV and a filtered embeddings NPZ.

Requirements:
    pip install numpy pillow requests

Usage:
    python sort_by_resolution.py --embeds my_embeds.npz --min_width 800 --min_height 600 --output_csv sorted_by_resolution.csv --output_npz my_embeds_filtered.npz

This script:
 1. Loads your embeddings NPZ (arrays: 'embeddings', 'urls', optional 'failed').
 2. Downloads each image URL, measures its resolution (width × height), and prints progress.
 3. Labels each entry as 'good' or 'bad' based on the thresholds.
 4. Writes a CSV with all URLs grouped: first all 'good', then all 'bad'.
 5. Filters the embeddings NPZ to keep only 'good' entries and saves a new NPZ.
"""
import argparse
import numpy as np
import csv
import sys
import requests
from PIL import Image
from io import BytesIO


def get_image_size(url, timeout=10):
    """Download an image and return its width and height. Returns (None, None) on failure."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        return img.width, img.height
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Filter by resolution and save CSV + NPZ of good-quality images.")
    parser.add_argument('--embeds', '-e', required=True,
                        help='Path to embeddings NPZ (e.g., my_embeds.npz)')
    parser.add_argument('--min_width',   type=int, default=800,
                        help='Minimum width for a good-quality image')
    parser.add_argument('--min_height',  type=int, default=600,
                        help='Minimum height for a good-quality image')
    parser.add_argument('--output_csv',  default='sorted_by_resolution.csv',
                        help='Output CSV file grouping good/bad URLs')
    parser.add_argument('--output_npz',  default='my_embeds_filtered.npz',
                        help='Output NPZ file with only good-quality embeddings')
    args = parser.parse_args()

    # Load embeddings and URLs
    try:
        data = np.load(args.embeds, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: embeddings file '{args.embeds}' not found.")
        sys.exit(1)
    embeddings = data['embeddings']
    urls       = data['urls']
    failed     = data['failed'] if 'failed' in data else []

    total = len(urls)
    print(f"Loaded {total} URLs from {args.embeds}")

    # Prepare lists
    good_entries = []  # tuples: (url, width, height)
    bad_entries  = []
    good_indices = []  # indices of good URLs

    # Process each URL
    for idx, url in enumerate(urls, start=1):
        w, h = get_image_size(url)
        is_good = (w or 0) >= args.min_width and (h or 0) >= args.min_height
        quality = 'good' if is_good else 'bad'
        print(f"[{idx}/{total}] {url} -> {w or 0}x{h or 0} => {quality}")

        if is_good:
            good_entries.append((url, w or 0, h or 0))
            good_indices.append(idx-1)
        else:
            bad_entries.append((url, w or 0, h or 0))

    # 4. Write CSV grouped by quality
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['url', 'width', 'height', 'quality'])
        # good first
        for url, w, h in good_entries:
            writer.writerow([url, w, h, 'good'])
        # then bad
        for url, w, h in bad_entries:
            writer.writerow([url, w, h, 'bad'])

    print(f"Written CSV: {len(good_entries)} good, {len(bad_entries)} bad -> {args.output_csv}")

    # 5. Filter NPZ and save
    if good_indices:
        filtered_embeddings = embeddings[good_indices]
        filtered_urls       = urls[good_indices]
        filtered_failed     = [(i, url) for i, url in failed if url in {u for u,_,_ in good_entries}]

        np.savez_compressed(
            args.output_npz,
            embeddings=filtered_embeddings,
            urls=filtered_urls,
            failed=np.array(filtered_failed, dtype=object)
        )
        print(f"Saved filtered NPZ: {len(filtered_urls)} embeddings -> {args.output_npz}")
    else:
        print("No good-quality images found; no NPZ saved.")

if __name__ == '__main__':
    main()
