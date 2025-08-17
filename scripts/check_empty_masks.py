
#!/usr/bin/env python3
"""
Find mask files that are completely empty (all zeros).
Requires: pip install imageio
"""

import os
import glob
import imageio.v3 as imageio
import numpy as np

def find_empty_masks(label_dir):
    empty = []
    for path in glob.glob(os.path.join(label_dir, "*.tif")):
        try:
            mask = imageio.imread(path)
            if not np.any(mask):
                empty.append(path)
        except Exception:
            pass
    return empty

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir", required=True)
    args = p.parse_args()
    empty = find_empty_masks(args.label_dir)
    if empty:
        print(f"Found {len(empty)} empty mask files:")
        for f in empty:
            print(f"  - {os.path.basename(f)}")
    else:
        print("Success! No empty masks were found.")
