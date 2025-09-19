#!/usr/bin/env python3
import glob
import re
import os

def sorted_failed_txts():
    files = glob.glob("valid*_failed.txt")
    def keyfn(f):
        m = re.search(r"valid(\d+)_failed\.txt$", os.path.basename(f))
        return int(m.group(1)) if m else f
    return sorted(files, key=keyfn)

def main():
    failed_files = sorted_failed_txts()
    if not failed_files:
        print("No 'valid#_failed.txt' files found in this directory.")
        return

    out_path = "valid_all_failed.txt"
    with open(out_path, "w", encoding="utf-8") as out_f:
        for fname in failed_files:
            with open(fname, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())
    print(f"Combined {len(failed_files)} files into '{out_path}'.")

if __name__ == "__main__":
    main()
