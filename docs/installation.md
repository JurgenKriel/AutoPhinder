# 1. Installation

## 1.1 IMOD

Manual segmentation of autophagosomes was performed using IMOD's **3dmod** application.

IMOD can be downloaded from the [IMOD download and documentation page](https://bio3d.colorado.edu/imod/).
The AutoPhinder workflow was developed using **IMOD v5.1.0**.

After installation, confirm that 3dmod opens correctly before proceeding.

## 1.2 Python environment

AutoPhinder was developed using **Python 3.11**.

We recommend creating a dedicated conda environment rather than installing the dependencies into the system
Python environment.

```bash
conda create -n autophinder python=3.11
conda activate autophinder
```

Install the required packages:

```bash
pip install torch torchvision
pip install micro-sam torch-em
pip install numpy imageio pillow scikit-image scikit-learn matplotlib
```

The data-preparation utilities in `scripts/` need a few extra helpers:

```bash
pip install -r requirements-extra.txt   # pillow, tifffile, tqdm, pandas, scikit-image
```

!!! note "PyTorch and GPUs"
    The exact PyTorch installation depends on whether you are using a CPU or an NVIDIA GPU. For GPU
    training, install the PyTorch build that matches your CUDA installation; see the
    [PyTorch installation selector](https://pytorch.org/get-started/locally/).

!!! tip "micro-SAM via conda"
    micro-SAM's own documentation recommends installing it from conda-forge
    (`conda install -c conda-forge micro_sam`), which also pulls in compatible versions of its
    dependencies. Either route works for AutoPhinder.

## 1.3 Check the GPU

To confirm that PyTorch can detect an NVIDIA GPU:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

A successful GPU setup returns `True` followed by the name of your GPU, for example:

```text
True
NVIDIA A100
```
