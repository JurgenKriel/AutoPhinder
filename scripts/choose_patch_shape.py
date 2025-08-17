
import imageio.v3 as imageio
import numpy as np

def choose_patch_shape(raw_paths):
    min_h, min_w = None, None
    for ip in raw_paths:
        arr = imageio.imread(ip)
        if arr.ndim > 2 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        h, w = arr.shape
        min_h = h if min_h is None else min(min_h, h)
        min_w = w if min_w is None else min(min_w, w)
    ph = max(min(512, min_h), 128)
    pw = max(min(512, min_w), 128)
    return (ph, pw)
