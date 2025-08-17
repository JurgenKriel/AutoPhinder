
#!/usr/bin/env python3
"""
Split multi-channel TIFFs into _c1.tif and _c2.tif stacks.
Requires: pip install tifffile
"""

import os
import re
import glob
import numpy as np
import tifffile as tiff

def list_input_tiffs(input_dir):
    files = []
    for p in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    files = sorted({f for f in files if os.path.isfile(f)})
    return [f for f in files if not re.search(r'_(?:c1|c2)\.(?:tif|tiff)$', os.path.basename(f), re.I)]

def determine_channel_axis(series, arr_shape):
    axes = getattr(series, 'axes', '') or ''
    if 'C' in axes:
        return axes.index('C'), axes
    if 'S' in axes:
        return axes.index('S'), axes
    candidates = [i for i, s in enumerate(arr_shape) if s in (2, 3, 4)]
    return (candidates[-1], axes) if candidates else (None, axes)

def split_file_into_c1_c2(tiff_path, output_dir, overwrite=False):
    base = os.path.splitext(os.path.basename(tiff_path))[0]
    out_c1 = os.path.join(output_dir, f"{base}_c1.tif")
    out_c2 = os.path.join(output_dir, f"{base}_c2.tif")

    if not overwrite and os.path.exists(out_c1) and os.path.exists(out_c2):
        print(f"Skipping (already split): {os.path.basename(tiff_path)}")
        return 0

    try:
        with tiff.TiffFile(tiff_path) as tf:
            series = tf.series[0]
            arr = series.asarray()
            chan_axis, _ = determine_channel_axis(series, arr.shape)
            if chan_axis is None:
                print(f"Could not determine channel axis for '{os.path.basename(tiff_path)}'; skipping.")
                return 0
            n_channels = arr.shape[chan_axis]
            if n_channels < 2:
                print(f"Only {n_channels} channel(s); skipping '{os.path.basename(tiff_path)}'.")
                return 0
            if n_channels > 2:
                print(f"Found {n_channels} channels; writing only the first two.")

            c1 = np.take(arr, 0, axis=chan_axis)
            c2 = np.take(arr, 1, axis=chan_axis)

            os.makedirs(output_dir, exist_ok=True)
            tiff.imwrite(out_c1, c1)
            tiff.imwrite(out_c2, c2)

        print(f"Wrote: {os.path.basename(out_c1)} and {os.path.basename(out_c2)}")
        return 2
    except Exception as e:
        print(f"Error processing '{os.path.basename(tiff_path)}': {e}")
        return 0

def split_multichannel_tiffs(input_dir='.', output_dir=None, overwrite=False):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    files = list_input_tiffs(input_dir)
    if not files:
        print("No candidate TIFF files found to split.")
        return

    print(f"Found {len(files)} TIFF file(s) to check:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    total_written = 0
    for f in files:
        print(f"\nProcessing: {os.path.basename(f)}")
        total_written += split_file_into_c1_c2(f, output_dir, overwrite=overwrite)

    print(f"\nDone. Channel-stacks written: {total_written}")

if __name__ == "__main__":
    split_multichannel_tiffs()
