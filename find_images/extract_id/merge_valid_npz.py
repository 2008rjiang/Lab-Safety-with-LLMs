#!/usr/bin/env python3
import numpy as np
import glob
import os
import re

def sorted_npz(files, prefix="valid"):
    """
    Sorts filenames like 'valid3.npz' or 'valid3_failed.npz' by the integer after 'valid'.
    """
    def keyfn(f):
        m = re.search(rf'{prefix}(\d+)', os.path.basename(f))
        return int(m.group(1)) if m else f
    return sorted(files, key=keyfn)

def merge_npz_list(file_list):
    """
    Given a list of .npz paths, loads each, groups arrays by key, 
    and concatenates them along axis 0.
    """
    accum = {}
    for path in file_list:
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                accum.setdefault(key, []).append(data[key])
    merged = {}
    for key, arrs in accum.items():
        try:
            merged[key] = np.concatenate(arrs, axis=0)
        except ValueError:
            # fallback for object‐dtype, etc.
            merged[key] = np.concatenate([a.astype(object) for a in arrs], axis=0)
    return merged

def main():
    # find and sort good vs. bad
    all_npz = glob.glob("valid*.npz")
    good_npz = [f for f in all_npz if not f.endswith("_failed.npz")]
    bad_npz  = [f for f in all_npz if    f.endswith("_failed.npz")]

    good_npz = sorted_npz(good_npz, prefix="valid")
    bad_npz  = sorted_npz(bad_npz,  prefix="valid")

    if not good_npz:
        print("No good .npz files found (pattern: valid#.npz).")
    else:
        print("Merging good files:", good_npz)
        good_merged = merge_npz_list(good_npz)
        np.savez_compressed("valid_all.npz", **good_merged)
        print("→ Saved all good links to valid_all.npz")

    if not bad_npz:
        print("No bad .npz files found (pattern: valid#_failed.npz).")
    else:
        print("Merging bad files:", bad_npz)
        bad_merged = merge_npz_list(bad_npz)
        np.savez_compressed("valid_all_failed.npz", **bad_merged)
        print("→ Saved all failures to valid_all_failed.npz")

if __name__ == "__main__":
    main()
