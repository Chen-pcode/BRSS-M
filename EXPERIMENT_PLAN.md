# BRSS-MambaSeg Experiment Plan

## Hypothesis

At low resolution, an official Mamba selective state-space block can capture
lesion-scale context without high-resolution attention cost. A row-major Raster
sequence provides a lightweight global context path, while multi-scale
boundary supervision should improve final contour generalization on an
external domain.

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
| BRSS-BCR-Mamba | Full six-level model with compressed shared row/column Mamba at 16 x 16 | Reference result |
| w/o Mamba | CNN-only encoder at the deep stages | Value of Mamba global modeling |
| w/o compression | Keep full Mamba channel width | Value of channel compression |
| single-axis Mamba | Use row-major Mamba only | Value of shared row/column modeling |
| w/o boundary modulation | Remove boundary-conditioned input modulation | Value of boundary-aware propagation |
| Final boundary supervision only | Remove deep boundary supervision | Multi-scale structural supervision |
| w/o boundary loss | Keep architecture, remove all boundary loss | Objective-level contribution |

The legacy `brss_raster_mamba` name is retained only for compatibility with
older checkpoints and refers to a single-axis, non-boundary-modulated model;
it is not part of the new primary ablation table.

Do not claim a component improves performance unless its three-seed mean and
paired per-image Dice comparison are consistent. Report both segmentation and
boundary metrics for the boundary-supervision claim. All models must use the
same split manifests and protocol-specific checkpoint selection.
