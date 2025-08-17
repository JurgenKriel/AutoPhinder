
import numpy as np

def preprocess_for_sam(img):
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_8 = (img_norm * 255).astype(np.uint8)
    return np.stack((img_8,) * 3, axis=-1) if img_8.ndim == 2 else img_8
