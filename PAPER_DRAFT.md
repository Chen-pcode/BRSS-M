# Lightweight Raster Mamba for Cross-Dataset Skin Lesion Segmentation

## 1. Paper Positioning

This paper studies whether a lightweight Raster Mamba encoder can improve skin
lesion segmentation while preserving a very small computational footprint. The
final model is `brss_raster_final_boundary`: a six-stage depthwise residual
U-Net, one uncompressed row-major official Mamba block at the 32 x 32 encoder
feature level, and a final decoder boundary prediction. The method is not
presented as a new Mamba state-space equation; its contribution is a careful,
efficient placement and an experimentally validated segmentation design.

The decoder mask-guided Mamba bridge was explored but is not the final method:
its three-seed results were less stable than the simpler Raster Mamba baseline.

## 2. Research Problem

Skin lesion segmentation requires both global lesion-shape context and accurate
local boundaries. CNNs provide efficient local extraction but have limited
effective context. Mamba offers linear-complexity sequence modeling, but its
benefit depends strongly on scan layout, feature resolution, and model width.
The unresolved practical question is therefore:

> Can a very small Raster Mamba module, placed at a resolution that retains
> enough spatial detail, improve cross-dataset segmentation without the cost and
> instability of multi-stage or heavily customized Mamba blocks?

## 3. Contributions

1. We develop a lightweight six-stage U-Net-style segmenter with one official
   Raster Mamba block at the 32 x 32 feature level.
2. We use final boundary supervision to improve the contour-sensitive output
   without adding unreliable multi-scale boundary objectives.
3. We conduct controlled placement and decoder-side Mamba studies, including
   CNN-only, encoder Raster Mamba, decoder Mamba, mask-guided fusion, uniform
   mask control, and 16 x 16 bridge control.
4. We report cross-dataset Dice, IoU, accuracy, sensitivity, specificity, HD95,
   parameters, FLOPs, model size, and runtime under a reproducible protocol.

## 4. Method

### 4.1 Backbone

The input is resized to 256 x 256. The encoder contains six resolution levels
with depthwise residual blocks and channel widths `[16, 16, 32, 48, 64, 96]`.
The decoder progressively upsamples and fuses skip features with a boundary
prediction branch.

### 4.2 Raster Mamba Context Block

The 32 x 32 encoder feature is flattened in row-major order into a sequence of
1024 tokens and processed by one official `mamba-ssm` block. The feature is
returned through a residual projection. This placement exposes a substantially
longer sequence than 16 x 16 or 8 x 8 while avoiding the cost of applying Mamba
at multiple resolutions.

### 4.3 Boundary Objective

The main segmentation output uses BCE plus Dice loss. A final decoder boundary
prediction is trained with an auxiliary binary cross-entropy term. Multi-scale
boundary supervision is excluded from the final configuration because the
controlled experiments did not show a consistent benefit.

## 5. Experimental Protocol

- Training data: ISIC2018, with separate validation and cross-dataset testing.
- Evaluation data: ISIC2018 validation, ISIC2017 validation, and PH2 test.
- Input: 256 x 256 RGB.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Batch size: 16.
- Maximum epochs: 300; early stopping patience: 60.
- Seeds: 42, 1234, and 2026.
- Main metrics: Dice and HD95. Secondary metrics: IoU, accuracy,
  sensitivity, specificity.

## 6. Three-Seed Results Currently Available

The following values are means over seeds 42, 1234, and 2026. They are based
on the current ISIC2018-training protocol and should be recomputed by the final
paper table script after all run folders are archived.

| Model | ISIC2018 Dice | ISIC2017 Dice | PH2 Dice | PH2 HD95 |
| --- | ---: | ---: | ---: | ---: |
| CNN-only + final boundary | 0.8863 | 0.8861 | **0.9177** | **13.46** |
| Raster Mamba + final boundary | **0.8864** | **0.8896** | 0.9166 | 13.89 |
| Decoder Mamba bridge | 0.8851 | 0.8889 | 0.9162 | 13.62 |
| Mask-guided fusion | 0.8858 | 0.8837 | 0.9110 | 14.53 |
| MGMB bridge | 0.8848 | 0.8855 | 0.9154 | 14.61 |

Raster Mamba is the most balanced Mamba candidate: it obtains the best mean
Dice on both ISIC evaluation sets, with 0.136M parameters and 0.641G estimated
FLOPs. CNN-only remains strongest on mean PH2 Dice and PH2 HD95, so the paper
must not claim universal superiority.

## 7. Required Before Submission

1. Re-run the final model and CNN baseline from clean archived folders and
   generate one reproducible three-seed summary.
2. Train on ISIC2017 as a second source domain, then evaluate on ISIC2017,
   ISIC2018, and PH2.
3. Add fair comparisons with U-Net, UNet++, Attention U-Net, EGE-UNet,
   UltraLight VM-UNet, and recent lightweight Mamba models.
4. Measure batch-1 GPU inference latency/FPS on the same T4 environment.
5. Run paired per-image Dice tests and report confidence intervals or Wilcoxon
   signed-rank p-values.
6. Add qualitative results and failure cases for low contrast, hair artifacts,
   irregular boundaries, and small lesions.

## 8. Honest Conclusion Template

The experiments indicate that a single carefully placed Raster Mamba block is
more stable than the tested decoder mask-guided alternatives and provides a
useful global-context module at a very small computational cost. Its advantage
is dataset-dependent rather than universal: CNN-only is competitive on PH2,
while Raster Mamba is stronger on the ISIC evaluation sets. This supports a
lightweight and reproducible Mamba design, but not a claim that Mamba always
outperforms convolutional segmentation.
