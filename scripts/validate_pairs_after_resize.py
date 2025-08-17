
#!/usr/bin/env python3
"""
Validate that, after resizing labels to image shape, masks still contain viable instances.
Requires: pip install imageio scikit-image tqdm pandas
"""

import os
import glob
import re
from pathlib import Path
import numpy as np
import imageio.v3 as imageio
from skimage.transform import resize as sk_resize
from tqdm import tqdm

def canonical_stem(p: Path) -> str:
    stem = p.stem.lower()
    stem = re.sub(r"[ _]+", "_", stem)
    stem = re.sub(r"(?:_mask|_masks|_label|_labels|_seg|_segmentation|_gt|_transformed)$", "", stem)
    stem = stem.replace("_transformed_", "_")
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem

def read_2d(path: str) -> np.ndarray:
    arr = imageio.imread(path)
    if arr.ndim > 2 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D image for {path}, got shape {arr.shape}")
    return arr

def resize_label_to_image(lbl: np.ndarray, img_shape: tuple) -> np.ndarray:
    if lbl.shape == img_shape:
        return lbl
    out = sk_resize(lbl, img_shape, order=0, preserve_range=True, anti_aliasing=False)
    out = np.rint(out)
    out = out.astype(lbl.dtype) if np.issubdtype(lbl.dtype, np.integer) else out.astype(np.int32)
    return out

def collect_paired_files(image_dir: str, label_dir: str):
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    lbl_paths = sorted(glob.glob(os.path.join(label_dir, "*.tif")))
    lbl_map = {canonical_stem(Path(lp)): lp for lp in lbl_paths}
    matched_pairs = []
    for ip in img_paths:
        key = canonical_stem(Path(ip))
        if key in lbl_map:
            matched_pairs.append((ip, lbl_map[key]))
    return matched_pairs

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--min_instance_size", type=int, default=25)
    args = p.parse_args()

    pairs = collect_paired_files(args.image_dir, args.label_dir)
    print(f"Found {len(pairs)} name-matched image/label pairs.")

    problematic = []
    for img_path, lbl_path in tqdm(pairs, desc="Analyzing paired files"):
        try:
            image = read_2d(img_path)
            label = read_2d(lbl_path)
            processed_label = resize_label_to_image(label, image.shape)
            ids = np.unique(processed_label)
            ids = ids[ids != 0]
            if len(ids) == 0:
                problematic.append((os.path.basename(img_path), os.path.basename(lbl_path), "no_objects", 0))
                continue
            max_size = max(np.sum(processed_label == i) for i in ids)
            if max_size < args.min_instance_size:
                problematic.append((os.path.basename(img_path), os.path.basename(lbl_path), "too_small", int(max_size)))
        except Exception as e:
            problematic.append((os.path.basename(img_path), os.path.basename(lbl_path), f"error:{e}", 0))

    if problematic:
        print(f"\nFound {len(problematic)} problematic file pairs:")
        for rec in problematic:
            print(f"  image={rec[0]}  label={rec[1]}  reason={rec[2]}  max_instance_size={rec[3]}")
        print("\nRecommendation: move/delete these LABELs or adjust min_instance_size.")
    else:
        print("\nSuccess! All processed masks contain at least one instance of sufficient size.")
