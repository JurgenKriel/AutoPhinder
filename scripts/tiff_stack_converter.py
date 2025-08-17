
#!/usr/bin/env python3
"""
Convert TIFF stacks to individual TIFF frames.
Requires: pip install pillow
"""

import os
import glob
from PIL import Image

def convert_tiff_stack_to_individual_images(input_dir='.', output_dir='converted_images', pattern="*_c1.tif"):
    os.makedirs(output_dir, exist_ok=True)
    tiff_files = glob.glob(os.path.join(input_dir, pattern))
    if not tiff_files:
        print(f"No TIFF stacks found matching {pattern} in {input_dir}")
        return
    total = 0
    for tiff_path in sorted(tiff_files):
        base = os.path.splitext(os.path.basename(tiff_path))[0]
        try:
            with Image.open(tiff_path) as img:
                i = 0
                while True:
                    frame_path = os.path.join(output_dir, f"{base}_frame_{i:03d}.tif")
                    img.save(frame_path)
                    total += 1
                    i += 1
                    try:
                        img.seek(img.tell() + 1)
                    except EOFError:
                        break
            print(f"Extracted frames from {os.path.basename(tiff_path)}")
        except Exception as e:
            print(f"Error processing {tiff_path}: {e}")
    print(f"Done. Extracted {total} frames into {output_dir}")

if __name__ == "__main__":
    convert_tiff_stack_to_individual_images()
