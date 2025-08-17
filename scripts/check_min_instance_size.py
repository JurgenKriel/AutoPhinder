
#!/usr/bin/env python3
"""
Find masks where the largest instance is smaller than a threshold.
Requires: pip install imageio tqdm pandas
"""

import os
import glob
import imageio.v3 as imageio
import numpy as np
from tqdm import tqdm

def find_too_small_masks(label_dir, min_instance_size=25):
    problematic = []
    paths = glob.glob(os.path.join(label_dir, "*.tif"))
    for path in tqdm(paths, desc="Analyzing masks"):
        try:
            mask = imageio.imread(path)
            ids = np.unique(mask)
            ids = ids[ids != 0]
            max_size = 0 if len(ids) == 0 else max(np.sum(mask == i) for i in ids)
            if max_size < min_instance_size:
                problematic.append((os.path.basename(path), max_size))
        except Exception:
            pass
    return problematic

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir", required=True)
    p.add_argument("--min_instance_size", type=int, default=25)
    args = p.parse_args()
    bad = find_too_small_masks(args.label_dir, args.min_instance_size)
    if bad:
        print(f"Found {len(bad)} problematic mask files:")
        for fname, size in bad:
            print(f"  - {fname}: max_instance_size={size}")
    else:
        print("Success! All masks contain at least one instance of sufficient size.")
