# BRSS-MambaSeg Experiment Plan

## Hypothesis

At low resolution, official Mamba selective state-space blocks can capture
lesion-scale context without high-resolution attention cost. Bidirectional
axial tokenization should preserve two-dimensional context, while a local
convolutional path retains contour and texture evidence. Multi-scale boundary
supervision should improve final contour generalization on an external domain.

## Fixed Protocol

- Input: 256 x 256 RGB images; the same augmentation, optimizer, scheduler,
  epoch budget, checkpoint rule, threshold and seeds for every variant.
- Source domains: train separately on ISIC2017 and ISIC2018.
- Evaluation: source validation, the other ISIC validation set, and PH2.
- Repetitions: seeds 42, 1234 and 2026 for the proposed model and all ablations.
- Primary metrics: Dice and HD95. Secondary metrics: IoU and accuracy.
- Stratified analysis: small lesions, low-contrast lesions, and artifact-heavy
  images, defined before inspecting model outcomes.

## Ablation Table

| Variant | Tests | Expected evidence |
| --- | --- | --- |
| BRSS-MambaSeg | Full six-level model | Reference result |
| Raster Mamba | Use a single row-major Mamba scan | Value of bidirectional axial tokenization |
| w/o Mamba | CNN-only encoder at the deep stages | Value of Mamba global modeling |
| 5-stage | Remove the 8 x 8 level | Value of the deepest level |
| w/o local path | Remove depthwise local feature path | Local-global complementarity |
| Final boundary supervision only | Remove deep boundary supervision | Multi-scale structural supervision |
| w/o boundary loss | Keep architecture, remove all boundary loss | Objective-level contribution |

Do not claim a component improves performance unless its three-seed mean and
paired per-image Dice comparison are consistent. Report both segmentation and
boundary metrics for the boundary-supervision claim. All models must use the
same split manifests and protocol-specific checkpoint selection.
