#!/usr/bin/env python3
"""
Filter out bad-quality images from an embeddings NPZ, keeping only 'good' ones based on a resolution CSV.

Usage:
    python filter_good_images.py --embeds my_embeds.npz --resolution_csv sorted_by_resolution.csv --output my_embeds_filtered.npz

This script:
 1. Loads your embeddings NPZ (with arrays 'embeddings', 'urls', optionally 'failed').
 2. Reads the resolution CSV to identify which URLs are labeled 'good'.
 3. Filters the embeddings, URLs, and failures to keep only 'good' entries.
 4. Saves a new compressed NPZ with only the good-quality images.
"""
import argparse
import numpy as np
import csv
import sys


def main():
    parser = argparse.ArgumentParser(description="Filter embeddings NPZ to keep only good-quality images.")
    parser.add_argument('--embeds', '-e', required=True,
                        help='Path to embeddings .npz file (e.g., my_embeds.npz)')
    parser.add_argument('--resolution_csv', '-r', required=True,
                        help='CSV listing url,width,height,quality')
    parser.add_argument('--output', '-o', default='my_embeds_filtered.npz',
                        help='Output filtered .npz file')
    args = parser.parse_args()

    # 1. Load embeddings
    try:
        data = np.load(args.embeds, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: embeddings file '{args.embeds}' not found.")
        sys.exit(1)
    embeddings = data['embeddings']
    urls       = data['urls']
    failed     = data['failed'] if 'failed' in data else []

    # 2. Read resolution CSV and collect good URLs
    good_urls = set()
    try:
        with open(args.resolution_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            # Expect header: url,width,height,quality
            for row in reader:
                if len(row) >= 4 and row[3].strip().lower() == 'good':
                    good_urls.add(row[0])
    except FileNotFoundError:
        print(f"Error: resolution CSV '{args.resolution_csv}' not found.")
        sys.exit(1)

    # 3. Determine indices of good URLs
    mask = np.array([u in good_urls for u in urls])
    good_indices = np.where(mask)[0]

    # 4. Filter arrays
    filtered_embeddings = embeddings[good_indices]
    filtered_urls       = urls[good_indices]
    # Filter failed entries if present
    filtered_failed = [(i, url) for i, url in failed if url in good_urls]

    # 5. Save filtered NPZ
    np.savez_compressed(
        args.output,
        embeddings=filtered_embeddings,
        urls=filtered_urls,
        failed=np.array(filtered_failed, dtype=object)
    )

    print(f"Saved {len(filtered_urls)} good-quality embeddings to '{args.output}'.")
    if filtered_failed:
        print(f"Note: {len(filtered_failed)} failed downloads correspond to good URLs.")

if __name__ == '__main__':
    main()
