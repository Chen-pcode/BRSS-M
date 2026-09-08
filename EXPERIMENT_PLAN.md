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

## Decoder-Bridge Study

The high-resolution encoder HGM study did not establish a stable advantage
over CNN-only segmentation. The next study therefore removes encoder Mamba and
tests whether decoder-side semantic localization makes state-space context more
selective. A 16 x 16 decoder feature predicts a coarse lesion map, which is
upsampled to guide the following 32 x 32 cross-scale fusion. The full bridge
softly separates lesion and background streams, scans both with a shared
Raster Mamba, and fuses them through a residual projection. The predicted map,
not a ground-truth mask, is used at both training and inference.

| Variant | Tests | Fixed loss |
| --- | --- | --- |
| CNN-only final-boundary | Strong no-Mamba baseline | Segmentation plus final boundary loss |
| Encoder Raster Mamba final-boundary | Existing Mamba baseline | Segmentation plus final boundary loss |
| Decoder Mamba bridge | Decoder placement without mask conditioning | Segmentation plus final boundary loss |
| Mask-guided fusion | Coarse-mask grouping without Mamba | Segmentation plus final boundary loss |
| Full MGMB | Mask-conditioned shared lesion/background Mamba bridge | Segmentation plus final boundary loss |
| Uniform-mask MGMB | Replace the predicted spatial mask with a constant 0.5 map while retaining two streams and Mamba | Segmentation plus final boundary loss |
| MGMB at 16 x 16 | Move the complete bridge to the first 16 x 16 decoder fusion | Segmentation plus final boundary loss |

The full model is supported only when it exceeds both Mamba and CNN baselines
over three seeds, particularly on PH2 Dice and HD95, without a material drop
on either ISIC evaluation set.


Do not claim a component improves performance unless its three-seed mean and
paired per-image Dice comparison are consistent. Report both segmentation and
boundary metrics for the boundary-supervision claim. All models must use the
same split manifests and protocol-specific checkpoint selection.
