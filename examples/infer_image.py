
#!/usr/bin/env python3
"""
Example: robust AIS call compatible across micro_sam versions.
"""

import imageio.v3 as imageio
from sam_finetuning_clean import run_automatic_instance_segmentation

if __name__ == "__main__":
    image = imageio.imread("/path/to/test_image.tif")
    prediction = run_automatic_instance_segmentation(
        image=image,
        image_path=None,
        output_folder=None,
        checkpoint_path="./models/checkpoints/autophagosomes_vitb/best.pt",
        model_type="vit_b",
        device="cuda",
        tile_shape=None,   # e.g., (384, 384) for tiling
        halo=None          # e.g., (64, 64) for seamless stitching
    )
    print("Prediction shape:", prediction.shape, "dtype:", prediction.dtype)
