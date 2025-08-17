
#!/usr/bin/env python3
"""
Frame-aware image/mask pairing and cleaning.
Requires: pip install pillow
"""

import os
import re
import shutil
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

def extract_frame_index(name: str) -> Optional[int]:
    m = re.search(r'(?:^|[_\-\s])(frame|Frame)[_\-\s]?(\d+)(?:\D|$)', name)
    if m:
        return int(m.group(2))
    m = re.search(r'(?:^|[_\-\s])(z|Z|slice|Slice)[_\-\s]?(\d+)(?:\D|$)', name)
    return int(m.group(2)) if m else None

def first_integer_key(name: str) -> Optional[int]:
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else None

def parse_key(name: str) -> Optional[Tuple[int, int]]:
    obj = first_integer_key(name)
    if obj is None:
        return None
    frame = extract_frame_index(name) or 0
    return (obj, frame)

def collect_by_key(directory: str, exts=(".tif", ".tiff")) -> Dict[Tuple[int, int], List[str]]:
    mapping: Dict[Tuple[int, int], List[str]] = {}
    for fname in os.listdir(directory):
        if not fname.lower().endswith(exts):
            continue
        key = parse_key(fname)
        if key is None:
            continue
        mapping.setdefault(key, []).append(os.path.join(directory, fname))
    return mapping

def is_mask_empty(mask_path: str, threshold: int = 0) -> bool:
    try:
        with Image.open(mask_path) as im:
            arr = np.array(im.convert('L'))
        return not (arr > threshold).any()
    except Exception:
        return True

def same_size(path_a: str, path_b: str) -> bool:
    try:
        with Image.open(path_a) as ia, Image.open(path_b) as ib:
            return ia.size == ib.size
    except Exception:
        return False

def filter_dataset(
    images_dir: str,
    masks_dir: str,
    out_images_dir: str,
    out_masks_dir: str,
    mask_threshold: int = 0,
    dry_run: bool = True,
    verbose: bool = True
) -> dict:
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    img_map = collect_by_key(images_dir)
    mask_map = collect_by_key(masks_dir)

    kept_pairs: List[Tuple[str, str]] = []
    skipped_empty: List[Tuple[str, str]] = []
    skipped_size: List[Tuple[str, str]] = []
    skipped_missing: List[Tuple[int, int]] = []
    skipped_ambiguous: List[Tuple[int, int]] = []

    all_keys = sorted(set(img_map.keys()) | set(mask_map.keys()))

    if verbose:
        print(f"Found {len(img_map)} image keys and {len(mask_map)} mask keys")
        print(f"Evaluating {len(all_keys)} unique keys")

    for k in all_keys:
        img_paths = img_map.get(k, [])
        mask_paths = mask_map.get(k, [])

        if not img_paths or not mask_paths:
            skipped_missing.append(k)
            continue

        if len(img_paths) > 1 or len(mask_paths) > 1:
            if verbose:
                print(f"[AMBIGUOUS] Key {k}: {len(img_paths)} images, {len(mask_paths)} masks")
            skipped_ambiguous.append(k)
            continue

        img_path = img_paths[0]
        mask_path = mask_paths[0]

        if not same_size(img_path, mask_path):
            if verbose:
                print(f"[SIZE MISMATCH] {os.path.basename(img_path)} vs {os.path.basename(mask_path)}")
            skipped_size.append((img_path, mask_path))
            continue

        if is_mask_empty(mask_path, threshold=mask_threshold):
            if verbose:
                print(f"[EMPTY MASK] {os.path.basename(mask_path)} > threshold {mask_threshold}")
            skipped_empty.append((img_path, mask_path))
            continue

        if dry_run:
            if verbose:
                print(f"[KEEP] {os.path.basename(img_path)}  <->  {os.path.basename(mask_path)}")
        else:
            shutil.copy2(img_path, os.path.join(out_images_dir, os.path.basename(img_path)))
            shutil.copy2(mask_path, os.path.join(out_masks_dir, os.path.basename(mask_path)))
        kept_pairs.append((img_path, mask_path))

    summary = {
        "kept_count": len(kept_pairs),
        "skipped_empty_count": len(skipped_empty),
        "skipped_size_count": len(skipped_size),
        "skipped_missing_count": len(skipped_missing),
        "skipped_ambiguous_count": len(skipped_ambiguous),
        "out_images_dir": out_images_dir,
        "out_masks_dir": out_masks_dir,
        "dry_run": dry_run,
        "mask_threshold": mask_threshold,
    }

    if verbose:
        print("\nSummary:")
        print(f"  Kept pairs: {summary['kept_count']}")
        print(f"  Skipped (empty masks): {summary['skipped_empty_count']}")
        print(f"  Skipped (size mismatch): {summary['skipped_size_count']}")
        print(f"  Skipped (missing counterpart): {summary['skipped_missing_count']}")
        print(f"  Skipped (ambiguous matches): {summary['skipped_ambiguous_count']}")
        if dry_run:
            print("\nDry run mode: no files were copied. Set dry_run=False to apply.")

    return summary

if __name__ == "__main__":
    # Minimal CLI usage example; customize as needed or import in notebooks/scripts.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--masks_dir", required=True)
    p.add_argument("--out_images_dir", required=True)
    p.add_argument("--out_masks_dir", required=True)
    p.add_argument("--mask_threshold", type=int, default=0)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    filter_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        out_images_dir=args.out_images_dir,
        out_masks_dir=args.out_masks_dir,
        mask_threshold=args.mask_threshold,
        dry_run=not args.apply,
        verbose=True
    )
