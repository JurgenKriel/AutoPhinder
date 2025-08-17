import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import os
import re
import argparse
import inspect
from glob import glob
from pathlib import Path
from typing import Union, Tuple, Optional, List

import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.transform import resize as sk_resize

import torch
from torch.utils.data import DataLoader, Dataset

# Import what we can from torch_em
try:
    from torch_em.util.util import get_random_colors
    # The TensorboardLogger import has been removed, as it's handled automatically.
except ImportError:
    # Fallback color function
    def get_random_colors(labels):
        import matplotlib.pyplot as plt
        return plt.cm.tab20

import micro_sam.training as sam_training
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)

import gc
from micro_sam.training import train_sam
from micro_sam.util import get_sam_model


class ShuffleableDataLoader(DataLoader):
    """A simple DataLoader wrapper that exposes the 'shuffle' attribute."""
    def __init__(self, *args, **kwargs):
        # Store the shuffle argument as an attribute for the trainer to find
        self.shuffle = kwargs.get("shuffle", False)
        super().__init__(*args, **kwargs)


class SimpleImageDataset(Dataset):
    """Simple dataset for loading image-label pairs."""
    
    def __init__(self, image_paths, label_paths, patch_shape=(512, 512), n_samples=1000):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_shape = patch_shape
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
        
    def load_image(self, path):
        """Load image and ensure it's in 8-bit format [0, 255]."""
        img = imageio.imread(path)
        
        if img.ndim > 2 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        
        if img.ndim != 2:
            raise RuntimeError(f"Expected 2D image for {path}, got shape {img.shape}")
        
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
            
        return img

    def load_label(self, path):
        """Load label image without normalization."""
        label = imageio.imread(path)
        if label.ndim > 2 and label.shape[-1] == 1:
            label = np.squeeze(label, axis=-1)
        if label.ndim != 2:
            raise RuntimeError(f"Expected 2D label for {path}, got shape {label.shape}")
        return label
    
    def extract_patch(self, image, label):
        """Extract random patch from image and label."""
        h, w = image.shape
        ph, pw = self.patch_shape
        
        if h < ph or w < pw:
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = image.shape
        
        start_h = np.random.randint(0, max(1, h - ph + 1))
        start_w = np.random.randint(0, max(1, w - pw + 1))
        
        image_patch = image[start_h:start_h + ph, start_w:start_w + pw]
        label_patch = label[start_h:start_h + ph, start_w:start_w + pw]
        
        if label_patch.shape != image_patch.shape:
            label_patch = sk_resize(
                label_patch, image_patch.shape, 
                order=0, preserve_range=True, anti_aliasing=False
            ).astype(label_patch.dtype)
        
        return image_patch, label_patch

    def __getitem__(self, idx):
        while True:
            pair_idx = np.random.randint(0, len(self.image_paths))
            
            try:
                image = self.load_image(self.image_paths[pair_idx])
                label = self.load_label(self.label_paths[pair_idx])
            except Exception as e:
                print(f"Warning: Skipping corrupt file pair: {self.image_paths[pair_idx]}. Error: {e}")
                continue

            if not np.any(label):
                continue

            max_patch_attempts = 5
            for _ in range(max_patch_attempts):
                image_patch, label_patch = self.extract_patch(image, label)
                
                if np.any(label_patch):
                    if image_patch.ndim == 2:
                        image_patch = image_patch[..., None]
                    image_tensor = torch.from_numpy(image_patch).float()
                    image_tensor = image_tensor.permute(2, 0, 1)

                    label_tensor = torch.from_numpy(label_patch).long()
                    label_tensor = label_tensor.unsqueeze(0)
                    
                    return image_tensor, label_tensor


def canonical_stem(p: Path) -> str:
    """Normalize stems so 'Autophagosome 5_frame_001' matches 'Autophagosome_5_transformed_frame_001'."""
    stem = p.stem.lower()
    stem = re.sub(r"[ _]+", "_", stem)
    stem = re.sub(r"(?:_mask|_masks|_label|_labels|_seg|_segmentation|_gt|_transformed)$", "", stem)
    stem = stem.replace("_transformed_", "_")
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def collect_paired_files(image_dir: str, label_dir: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Return matched image/label lists by canonical stem; also return dropped raw files with reasons."""
    img_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
    lbl_paths = sorted(glob(os.path.join(label_dir, "*.tif")))
    assert len(img_paths) > 0, f"No TIFF images found in {image_dir}"
    assert len(lbl_paths) > 0, f"No TIFF labels found in {label_dir}"

    lbl_map = {}
    for lp in lbl_paths:
        key = canonical_stem(Path(lp))
        if key not in lbl_map:
            lbl_map[key] = lp

    matched_raw, matched_lbl = [], []
    dropped = []
    for ip in img_paths:
        key = canonical_stem(Path(ip))
        candidate = lbl_map.get(key, None)
        if candidate is None:
            dropped.append((ip, "no matching label by normalized name"))
            continue
        matched_raw.append(ip)
        matched_lbl.append(candidate)

    return matched_raw, matched_lbl, dropped


def choose_patch_shape(raw_paths: List[str]) -> Tuple[int, int]:
    """Pick a safe 2D patch shape that fits all images, capped at 512."""
    if not raw_paths:
        raise ValueError("Cannot choose patch shape from an empty list of images. Please check the file paths and matching logic.")
    min_h, min_w = None, None
    for ip in raw_paths:
        arr = imageio.imread(ip)
        if arr.ndim > 2 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        h, w = arr.shape
        min_h = h if min_h is None else min(min_h, h)
        min_w = w if min_w is None else min(min_w, w)
    ph = min(512, min_h)
    pw = min(512, min_w)
    ph = max(ph, 64)
    pw = max(pw, 64)
    return (ph, pw)

def run_automatic_instance_segmentation(
    checkpoint_path: Union[os.PathLike, str],
    model_type: str = "vit_b",
    device: Optional[Union[str, torch.device]] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    image: Optional[np.ndarray] = None,
    image_path: Optional[str] = None,
    output_folder: Optional[str] = None,
):
    """
    Runs automatic instance segmentation, adapting to the installed micro_sam version.
    """
    if image is None and image_path is None:
        raise ValueError("Either 'image' (np.ndarray) or 'image_path' (str) must be provided.")

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint_path,
        device=device,
        is_tiled=(tile_shape is not None),
    )

    sig_params = inspect.signature(automatic_instance_segmentation).parameters
    ais_kwargs = dict(
        predictor=predictor,
        segmenter=segmenter,
        ndim=2,
        tile_shape=tile_shape,
        halo=halo,
    )

    if "image" in sig_params or "input" in sig_params:
        if image is None:
            image = imageio.imread(image_path)
        param_key = "image" if "image" in sig_params else "input"
        ais_kwargs[param_key] = image
        return automatic_instance_segmentation(**ais_kwargs)
    elif "image_paths" in sig_params and "output_folder" in sig_params:
        if image_path is None or output_folder is None:
            raise ValueError("'image_path' and 'output_folder' are required for this version of micro_sam.")
        ais_kwargs["image_paths"] = [image_path]
        ais_kwargs["output_folder"] = output_folder
        automatic_instance_segmentation(**ais_kwargs)
        prediction_files = glob(os.path.join(output_folder, "*"))
        if len(prediction_files) != 1:
            raise FileNotFoundError(f"Expected 1 prediction file in {output_folder}, but found {len(prediction_files)}.")
        return imageio.imread(prediction_files[0])
    elif "input_path" in sig_params and "output_path" in sig_params:
        if image_path is None or output_folder is None:
            raise ValueError("'image_path' and 'output_folder' are required for this version of micro_sam.")
        output_prediction_path = os.path.join(output_folder, os.path.basename(image_path))
        ais_kwargs["input_path"] = image_path
        ais_kwargs["output_path"] = output_prediction_path
        automatic_instance_segmentation(**ais_kwargs)
        return imageio.imread(output_prediction_path)
    else:
        raise RuntimeError(
            "The installed 'automatic_instance_segmentation' function has an unsupported signature. "
            f"Expected 'image', 'input', 'image_paths', or 'input_path' argument, but got: {list(sig_params.keys())}"
        )


def run_inference(image_path: str, checkpoint_path: str, model_type: str, device: str, output_dir: str):
    """Run inference on a single specified image."""
    results_dir = os.path.join(output_dir, "inference_results")
    os.makedirs(results_dir, exist_ok=True)

    temp_segmentation_dir = os.path.join(output_dir, "temp_segmentations")
    os.makedirs(temp_segmentation_dir, exist_ok=True)

    print(f"Processing image: {os.path.basename(image_path)}")
    
    # Create a unique sub-folder for this specific image's segmentation to avoid conflicts
    image_base_name = Path(image_path).stem
    image_specific_output_dir = os.path.join(temp_segmentation_dir, image_base_name)
    os.makedirs(image_specific_output_dir, exist_ok=True)
    
    image = imageio.imread(image_path)
    prediction = run_automatic_instance_segmentation(
        image=image,
        image_path=image_path,
        output_folder=image_specific_output_dir,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(image, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Input Image")
    ax[1].imshow(prediction, cmap=get_random_colors(prediction), interpolation="nearest")
    ax[1].axis("off")
    ax[1].set_title("Predictions (AIS)")
    
    output_filename = f"result_{image_base_name}.png"
    output_path = os.path.join(results_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved result to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a SAM model.")
    parser.add_argument("--data_folder", required=True, help="Path to the training data.")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--n_iterations", type=int, default=1000, help="Number of iterations per epoch.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--checkpoint_name", type=str, default="sam_fibsem", help="Name of the checkpoint.")
    parser.add_argument("--model_type", type=str, default="vit_b_em_organelles", help="The SAM model type.")
    parser.add_argument("--save_root", type=str, default=None, help="Root directory for saving checkpoints and logs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--inference_image", type=str, default=None, help="Path to a single image for inference after training.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_dir = '/vast/scratch/users/kriel.j/fibsem/cropped_autophagosomes/individual_tiffs/clean_tiffs/'
    label_dir = '/vast/scratch/users/kriel.j/fibsem/transformed_masks/split_channels/individual_tiffs/clean_masks/'

    raw_paths, lbl_paths, dropped = collect_paired_files(image_dir, label_dir)
    if dropped:
        print(f"Warning: Dropped {len(dropped)} files that could not be matched.")

    split_index = int(len(raw_paths) * 0.9)
    train_raw, val_raw = raw_paths[:split_index], raw_paths[split_index:]
    train_lbl, val_lbl = lbl_paths[:split_index], lbl_paths[split_index:]

    if not val_raw:
        print("Warning: Not enough data for a validation split. Using all data for training and validation.")
        val_raw, val_lbl = train_raw, train_lbl

    print(f"Data split: {len(train_raw)} training, {len(val_raw)} validation samples.")

    patch_shape = choose_patch_shape(raw_paths)
    print(f"Using dynamically determined patch shape: {patch_shape}")

    train_dataset = SimpleImageDataset(train_raw, train_lbl, patch_shape=patch_shape, n_samples=args.n_iterations * args.batch_size)
    train_loader = ShuffleableDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = SimpleImageDataset(val_raw, val_lbl, patch_shape=patch_shape, n_samples=100)
    val_loader = ShuffleableDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    total_iterations = args.n_epochs * args.n_iterations
    print(f"Starting training for {args.n_epochs} epochs ({total_iterations} total iterations).")

    train_sam(
        name=args.checkpoint_name,
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_iterations=total_iterations,
        device=device,
        save_root=args.save_root,
        lr=args.lr,
        with_segmentation_decoder=False,
        #patience=None,  # This disables early stopping
    )

    print("Training finished.")

    final_checkpoint = os.path.join(args.save_root, "checkpoints_hpc", args.checkpoint_name, "best.pt")

    if os.path.exists(final_checkpoint) and args.inference_image:
        if os.path.exists(args.inference_image):
            print(f"\n--- Running Inference on specified image: {args.inference_image} ---")
            inference_output_dir = os.path.join(args.save_root, "inference_output")
            os.makedirs(inference_output_dir, exist_ok=True)
            
            run_inference(
                image_path=args.inference_image,
                checkpoint_path=final_checkpoint,
                model_type=args.model_type,
                device=device,
                output_dir=inference_output_dir
            )
        else:
            print(f"Error: The specified inference image does not exist: {args.inference_image}")
    elif args.inference_image:
        print(f"Could not find final checkpoint at {final_checkpoint} to run inference.")


if __name__ == '__main__':
    main()
