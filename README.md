# AutoPhinder
## Instance segmentation of autophagosomes in fibsem datasets 

## Data Preparation
- Utilities to prepare and clean datasets:
  - scripts/split_multichannel_tiffs.py splits multi-channel TIFFs into _c1/_c2 stacks.
  - scripts/tiff_stack_converter.py converts multi-page TIFF stacks to individual frame TIFFs.
  - scripts/filter_dataset_pairs.py pairs images and masks with frame-awareness and copies only valid pairs to clean folders (use dry-run first, then apply).
  - scripts/check_empty_masks.py and scripts/check_min_instance_size.py detect empty masks or masks without sufficiently large instances; align the min size with your MinInstanceSampler(min_size).
  - scripts/validate_pairs_after_resize.py ensures that, after resizing labels to image shape, viable instances remain.

## Inference
- Preprocessing for SAM (microscopy TIFFs to 8-bit RGB) is provided in scripts/preprocess_for_sam.py. Use it if your micro_sam version supports ndarray input.
- The run_automatic_instance_segmentation wrapper in sam_finetuning_clean.py adapts to different micro_sam signatures (image/input vs input_path/output_path vs image_paths/output_folder). It supports tiling and halo to handle large images.

## Tips/Troubleshooting
- If predictions are empty, relax filtering like pred_iou_thresh and min_mask_region_area, double-check image preprocessing, and verify the checkpoint.
- Keep MinInstanceSampler(min_size) consistent with your data cleaning thresholds.

## Requirements (Optional)
- Optional helpers used by scripts are in requirements-extra.txt. Install with: pip install -r requirements-extra.txt
