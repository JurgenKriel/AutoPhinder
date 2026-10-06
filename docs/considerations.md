# Important considerations

## Model performance depends on the dataset

AutoPhinder was trained using FIB-SEM data containing autophagosomes with specific ultrastructural
characteristics. Performance may decrease when applying the model to:

- different cell types
- different fixation protocols
- different FIB-SEM acquisition settings
- different voxel sizes
- poor-quality datasets
- datasets containing substantially different ultrastructural morphology

For these datasets, additional training examples or further fine-tuning may improve performance: annotate a
small number of autophagosomes in your own data ([Ground truth in IMOD](ground-truth.md)) and fine-tune starting
from the AutoPhinder checkpoint.

## Automated segmentation should be validated

AutoPhinder is intended to reduce the time required for manual segmentation, but predictions should be
visually inspected ([Visualising the segmentation](inference.md#54-visualising-the-segmentation)) and, where
appropriate, compared against expert annotations and/or ground truth.

## Original AutoPhinder training dataset

The original AutoPhinder dataset was generated from manually segmented FIB-SEM/CLEM datasets
(~32 autophagosomes per cell, traced in IMOD).

More than **5,000 labelled examples** were generated from the manually segmented autophagosomes and
subsequently used for model training. Training was performed on an **NVIDIA A100** GPU in an HPC environment.
