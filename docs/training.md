# 4. Training AutoPhinder

!!! info "You do not need to retrain to use AutoPhinder"
    If you simply want to segment a new dataset, load the trained checkpoint and go to
    [Inference on new data](inference.md).

## 4.1 Pre-trained model

AutoPhinder uses the **micro-SAM ViT-B** model as its starting point.

Rather than training a segmentation model from scratch, the pre-trained model is fine-tuned on the manually
generated FIB-SEM autophagosome annotations. This allows the model to adapt its learned image features to the
ultrastructural characteristics of autophagosomes.

## 4.2 Training configuration

The original AutoPhinder training configuration:

| Parameter | Value |
|---|---|
| Model | micro-SAM ViT-B |
| Training epochs | 5 |
| Iterations / epoch | 1,000 |
| Batch size | 1 |
| Objects / batch | 5 |
| Train / validation split | 80 / 20 |
| Sampler | `MinInstanceSampler` |
| Training hardware | NVIDIA A100 |
| Training environment | HPC |

The model was trained end-to-end with an additional convolutional segmentation decoder.

## 4.3 Training data loader

Training data are loaded using torch-em and micro-SAM utilities.

The `default_sam_loader` is used in image-collection mode to avoid issues associated with stacking images of
different dimensions. The `MinInstanceSampler` ensures that sampled patches contain at least one foreground
object.

!!! important
    Many regions of a FIB-SEM dataset contain **no autophagosomes**. Without instance-focused sampling, the
    model would receive a large number of background-only patches during training.

## 4.4 Fine-tuning the model

Training is performed with `micro_sam.training.train_sam`, which handles:

- forward propagation through the model and prediction of segmentation masks
- calculation of segmentation and IoU-related losses
- backpropagation and optimisation of model parameters
- validation
- logging of training metrics

Logged metrics include `train/loss`, `train/model_iou` and validation metrics. They can be monitored during
training with TensorBoard:

```bash
tensorboard --logdir <save_root>/logs
```

## 4.5 Running training

The fine-tuning script in this repository is `fine_tune_vit_b.py`:

```bash
python fine_tune_vit_b.py \
    --data_folder /path/to/clean_dataset \
    --model_type vit_b \
    --n_epochs 5 \
    --n_iterations 1000 \
    --batch_size 1 \
    --save_root ./models \
    --checkpoint_name autophinder_model \
    --inference_image /path/to/test_image.tif   # optional: run AIS on one image when training ends
```

| Option | Default | Description |
|---|---|---|
| `--data_folder` | *(required)* | Path to the training data |
| `--n_epochs` | `50` | Number of epochs |
| `--n_iterations` | `1000` | Iterations per epoch |
| `--batch_size` | `1` | Batch size |
| `--model_type` | `vit_b_em_organelles` | micro-SAM model to start from |
| `--checkpoint_name` | `sam_fibsem` | Name of the checkpoint folder |
| `--save_root` | – | Root directory for checkpoints and logs |
| `--lr` | `1e-4` | Learning rate |
| `--inference_image` | – | Image to segment with the best checkpoint after training |

!!! warning "Current state of `fine_tune_vit_b.py`"
    The script in the repository is an HPC working copy and differs from the configuration above in a few
    places. Check these before you run it:

    - **Data paths are hard-coded.** `--data_folder` is parsed but not used; the image and label directories
      are set near the top of `main()`. Edit them to point at your curated `images/` and `masks/`.
    - **Validation split is 90/10**, not 80/20.
    - It uses a custom random-patch `SimpleImageDataset` (which skips empty patches) rather than
      `default_sam_loader` + `MinInstanceSampler`.
    - `with_segmentation_decoder=False`, so no extra decoder is trained. Set it to `True` to match the
      published configuration and to get the decoder-based AIS used at inference.
    - The defaults (`50` epochs, `vit_b_em_organelles`) differ from the published run (`5` epochs, `vit_b`),
      so pass them explicitly as in the example above.
    - After training it looks for the checkpoint in `<save_root>/checkpoints_hpc/`, but current micro-SAM
      releases save to `<save_root>/checkpoints/` (see below), so the optional `--inference_image` step may
      report that it cannot find the checkpoint.

    The [walkthrough notebook](notebooks/autophinder_walkthrough.ipynb) (section 5) shows the published
    configuration (`default_sam_loader` + `MinInstanceSampler`, segmentation decoder, 5 objects per batch)
    with the current micro-SAM API.

Before running the training script, check that:

- [ ] The training images are in the correct directory.
- [ ] The corresponding masks are present.
- [ ] Image and mask dimensions match.
- [ ] The Python environment is activated.
- [ ] PyTorch can access the GPU if GPU training is intended ([check](installation.md#13-check-the-gpu)).
- [ ] The required micro-SAM and torch-em versions are installed.

### On an HPC cluster (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=autophinder
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

conda activate autophinder
python fine_tune_vit_b.py --data_folder ... --save_root ./models --checkpoint_name autophinder_model
```

Adjust the partition and GPU names for your cluster.

## 4.6 Training output

After training, the best checkpoint is saved for subsequent inference:

```text
<save_root>/
└── checkpoints/
    └── autophinder_model/
        └── best.pt
```

The trained checkpoint can be loaded directly for inference. Do not retrain the model if you simply want to
use AutoPhinder on a new dataset.
