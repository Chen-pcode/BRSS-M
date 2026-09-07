# BRSS-MambaSeg Experiment Plan

## Hypothesis

Most lightweight skin-lesion Mamba networks apply Mamba only at 16 x 16 or
8 x 8 features, yielding short sequences that underuse its linear long-range
modeling advantage. Direct high-resolution Mamba is expensive because Mamba
cost increases with channel width. We therefore use grouped, channel-compressed
official Mamba at 32 x 32 features (1024 tokens), retaining long-range context
with low-dimensional shared state-space scans.

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
| HGM-Mamba (S3) | Full six-level model with compressed, grouped shared row/column Mamba at 32 x 32 | Reference result and high-resolution long-sequence modeling |
| HGM-Mamba (S3+S4) | Same HGM block at 32 x 32 and 16 x 16 | Value of a deep semantic supplement |
| HGM-Mamba (S3+S4+S5) | Same HGM block at 32 x 32, 16 x 16 and 8 x 8 | Whether progressive all-deep Mamba is beneficial |
| HGM-Mamba (S4 only) | Same HGM block at 16 x 16 only | Low-resolution Mamba placement baseline |
| Plain Raster Mamba | One uncompressed row-major Mamba scan at 32 x 32 | High-resolution Mamba baseline |
| w/o Mamba | CNN-only encoder at the deep stages | Value of Mamba global modeling |
| w/o compression | Keep full Mamba channel width | Value of channel compression |
| w/o grouping | Keep compressed channels in one Mamba sequence | Value of grouped shared scanning |
| single-axis Mamba | Use row-major Mamba only | Value of shared row/column modeling |
| Final boundary supervision only | Remove deep boundary supervision | Multi-scale structural supervision |
| w/o boundary loss | Keep architecture, remove all boundary loss | Objective-level contribution |


Do not claim a component improves performance unless its three-seed mean and
paired per-image Dice comparison are consistent. Report both segmentation and
boundary metrics for the boundary-supervision claim. All models must use the
same split manifests and protocol-specific checkpoint selection.
