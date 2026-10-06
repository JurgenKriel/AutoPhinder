# AutoPhinder
## Instance segmentation of autophagosomes in FIB-SEM datasets

AutoPhinder fine-tunes [micro-SAM](https://github.com/computational-cell-analytics/micro-sam) on manual IMOD
annotations to automatically segment autophagosomes in focused ion beam scanning electron microscopy (FIB-SEM) data.

**📖 Documentation: <https://jurgenkriel.github.io/AutoPhinder/>**: installation, ground-truth annotation in
IMOD, dataset curation, training, inference, and a full
[Jupyter walkthrough](docs/notebooks/autophinder_walkthrough.ipynb).

![AutoPhinder segmentation of autophagosomes](docs/assets/images/ais-results.png)

## Quick start

```bash
conda create -n autophinder python=3.11 && conda activate autophinder
pip install torch torchvision micro-sam torch-em numpy imageio pillow scikit-image scikit-learn matplotlib
pip install -r requirements-extra.txt
```

- **Train:** `python fine_tune_vit_b.py --help` ([Training guide](https://jurgenkriel.github.io/AutoPhinder/training/))
- **Segment new data:** `examples/infer_image.py` ([Inference guide](https://jurgenkriel.github.io/AutoPhinder/inference/))
- **Notebook:** `docs/notebooks/autophinder_walkthrough.ipynb`

## Repository layout

| Path | Contents |
|---|---|
| `fine_tune_vit_b.py` | micro-SAM fine-tuning + version-tolerant AIS wrapper |
| `examples/` | Inference example |
| `scripts/` | Data-preparation utilities (split/stack/pair/clean/check) |
| `docs/` | Documentation site source (MkDocs Material) and walkthrough notebook |

## Building the documentation locally

```bash
pip install -r docs-requirements.txt
mkdocs serve        # http://127.0.0.1:8000
```

The site is deployed to GitHub Pages by `.github/workflows/docs.yml` on every push to `main` that touches `docs/`.
