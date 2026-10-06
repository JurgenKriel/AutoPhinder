# Script reference

All scripts live in the [repository](https://github.com/JurgenKriel/AutoPhinder). Unless noted, they read and
write single-frame `.tif` files. Install their extra dependencies with `pip install -r requirements-extra.txt`.

## Notebook

| File | Purpose |
|---|---|
| `docs/notebooks/autophinder_walkthrough.ipynb` | End-to-end walkthrough: lazy loading, preprocessing, IMOD → masks, curation, fine-tuning, inference and quantification. [View it on the site](notebooks/autophinder_walkthrough.ipynb). |

## Training & inference

| File | Purpose |
|---|---|
| `fine_tune_vit_b.py` | Fine-tunes a micro-SAM model on paired images/masks and optionally runs AIS on one image afterwards. Also provides `run_automatic_instance_segmentation()`, a version-tolerant AIS wrapper. See [Training](training.md#45-running-training). |
| `examples/infer_image.py` | Minimal example: AIS on one image with a trained checkpoint. See [Inference](inference.md#53-automatic-instance-segmentation). |

## Data preparation — `scripts/`

### `split_multichannel_tiffs.py`
Splits multi-channel TIFFs into `<name>_c1.tif` and `<name>_c2.tif` stacks. The channel axis is taken from the
TIFF metadata (`C` or `S`), falling back to an axis of size 2–4. Already-split files are skipped.

### `tiff_stack_converter.py`
Converts multi-page TIFF stacks (default pattern `*_c1.tif`) into individual frames named
`<stack>_frame_NNN.tif` in `converted_images/`.

### `filter_dataset_pairs.py`
Frame-aware pairing of images and masks. Each file is keyed by its first integer (object ID) and a
`frame`/`z`/`slice` index; ambiguous keys, size mismatches and empty masks are dropped, and valid pairs are
copied to the output folders.

```bash
python scripts/filter_dataset_pairs.py --images_dir IMG --masks_dir MSK \
    --out_images_dir IMG_CLEAN --out_masks_dir MSK_CLEAN [--mask_threshold 0] [--apply]
```

Without `--apply` it only reports what it would do.

### `check_empty_masks.py`
Lists masks that are entirely zero.

```bash
python scripts/check_empty_masks.py --label_dir MSK
```

### `check_min_instance_size.py`
Lists masks whose largest instance is smaller than `--min_instance_size` pixels (default 25). Keep this
consistent with `MinInstanceSampler(min_size=...)`.

```bash
python scripts/check_min_instance_size.py --label_dir MSK --min_instance_size 25
```

### `validate_pairs_after_resize.py`
Pairs images and labels by a canonical stem (ignoring suffixes such as `_mask`, `_label`, `_gt`), resizes each
label to its image with nearest-neighbour interpolation and reports pairs left without a viable instance.

```bash
python scripts/validate_pairs_after_resize.py --image_dir IMG --label_dir MSK --min_instance_size 25
```

### `choose_patch_shape.py`
`choose_patch_shape(paths)` returns a training patch shape: the smallest image height/width in the dataset,
clamped to 128–512 px.

### `preprocess_for_sam.py`
`preprocess_for_sam(img)` min–max normalises an image to 8-bit and stacks greyscale to 3-channel RGB, the input
format SAM expects.

## Known gaps

!!! warning "Scripts referenced in the original wiki but not yet in the repository"
    - **`mask_processing.py`** (IMOD annotations → masks). The same conversion (`model2point` + per-slice
      `polygon2mask`) is implemented step by step in the
      [walkthrough notebook](notebooks/autophinder_walkthrough.ipynb), section 3.
    - **`train_model.py`**: training is done with `fine_tune_vit_b.py`.
    - **`sam_finetuning_clean.py`** (imported by `examples/infer_image.py`): the wrapper now lives in
      `fine_tune_vit_b.py`.
