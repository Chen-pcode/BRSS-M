# Raster Mamba with Boundary-Aware Decoding for Cross-Dataset Skin Lesion Segmentation

## Abstract

Accurate skin lesion segmentation requires both global lesion structure and
fine boundary detail. Convolutional encoders are effective for local feature
extraction, but their context is limited by stacked local operators. Mamba
provides linear-complexity sequence modeling, yet its benefit in medical image
segmentation depends strongly on scan order, feature resolution, and placement.
This work studies a controlled hybrid design that places one Raster Mamba block
at the 32 x 32 encoder feature level of a six-stage U-Net-like network. A final
decoder boundary objective is used to preserve contour information. Instead of
adding Mamba blocks at every resolution, we test whether one carefully selected
placement provides more reliable cross-dataset behavior. Experiments on
ISIC2017, ISIC2018, and PH2 report Dice, IoU, accuracy, sensitivity,
specificity, HD95, and computational cost. Three-seed results show that the
Raster Mamba model is competitive with a CNN-only baseline, obtains the best
mean Dice on the ISIC evaluation sets when trained on ISIC2018, and obtains
competitive PH2 performance when trained on ISIC2017. Decoder-side
mask-guided Mamba variants are not consistently superior, emphasizing the
importance of controlled placement and cross-dataset validation.

**Keywords:** skin lesion segmentation; Mamba; state-space model; Raster
Mamba; boundary supervision; cross-dataset generalization.

## 1. Introduction

Automated skin lesion segmentation is an important component of computer-aided
dermoscopic image analysis. Segmentation masks support lesion measurement,
feature extraction, diagnosis support, and longitudinal monitoring. However,
lesions vary in size, color, texture, shape, and boundary sharpness. Hair,
illumination variation, ruler marks, bubbles, and low contrast further complicate
the separation of lesion and surrounding skin.

U-Net-style convolutional architectures remain strong baselines because their
encoder-decoder structure combines semantic features with high-resolution skip
features. Skip connections are particularly important for dermoscopic images,
where boundaries may be thin, irregular, or visually similar to surrounding
skin. Nevertheless, convolution mainly aggregates information through local
neighborhoods. Enlarging context with deeper or wider convolutional blocks can
increase computation without providing an efficient long-range mechanism.

Vision Transformers model long-range interaction with self-attention, but the
quadratic token cost can be undesirable for high-resolution images. Mamba uses
selective state-space recurrence and linear sequence complexity as an
alternative. Its use in vision is not automatically beneficial: a two-
dimensional feature map must be serialized, and scan order determines how
spatially adjacent and distant pixels interact. Applying Mamba at multiple
resolutions can also add redundant computation and optimization instability.

This paper investigates the following question:

> Does one Raster Mamba block at a carefully selected intermediate resolution,
> combined with boundary-aware decoding, provide a more stable cross-dataset
> design than multi-stage or decoder-side Mamba alternatives?

The final model, BRSS-Raster, uses a six-stage convolutional encoder, one
official Raster Mamba block at the 32 x 32 feature level, progressive skip
fusion, and final boundary supervision. We do not claim a new state-space
equation. The contribution is a controlled study of Mamba placement in a skin
lesion segmenter, with explicit negative controls and two training-domain
protocols.

Our contributions are:

1. A controlled six-stage encoder-decoder with one Raster Mamba block at the
   intermediate 32 x 32 representation.
2. A final decoder boundary objective that adds contour supervision without
   relying on the multi-scale boundary constraints rejected by preliminary
   controls.
3. A systematic comparison of CNN-only, encoder Raster Mamba, decoder Mamba,
   mask-guided fusion, uniform-mask control, and a 16 x 16 bridge control.
4. Cross-dataset evaluation on ISIC2017, ISIC2018, and PH2 with segmentation,
   boundary, statistical, and efficiency metrics.

## 2. Related Work

### 2.1 CNN-Based Lesion Segmentation

Encoder-decoder networks use the encoder to construct semantic representations
and the decoder to recover spatial detail. Attention, group aggregation,
boundary losses, and multi-scale fusion have been used to improve lesion
segmentation. These designs remain largely convolutional and therefore depend
on stacked local operators for long-range context.

### 2.2 Mamba for Vision

Mamba uses selective state-space recurrence to model long sequences with linear
complexity. Vision Mamba methods adapt the operator by scanning image tokens in
one or more spatial directions. Multi-directional scanning can reduce
directional bias but increases implementation complexity. The resolution of the
scan is also important: low-resolution features are semantically strong but
contain fewer tokens and less boundary detail, whereas high-resolution features
provide longer sequences but are harder to optimize.

### 2.3 Motivation for Raster Placement

We use a row-major Raster scan at 32 x 32. For a 256 x 256 input this produces
1024 tokens, longer than 16 x 16 or 8 x 8 while remaining at an intermediate
encoder stage. The design is intentionally simple so that the effect of Mamba
can be separated from multi-directional scans, channel grouping, and
mask-conditioned branches.

## 3. Proposed Method

### 3.1 Overall Architecture

The input is resized to 256 x 256 and passed through a six-stage encoder-decoder.
The encoder uses depthwise residual convolutional blocks with channel widths
`[16, 16, 32, 48, 64, 96]`. Downsampling produces 256 x 256, 128 x 128,
64 x 64, 32 x 32, 16 x 16, and 8 x 8 representations. The 32 x 32 encoder
feature is processed by the only Mamba block in BRSS-Raster. The decoder
progressively upsamples and fuses skip features. A boundary prediction is
generated at each decoder fusion, while the final decoder feature produces the
segmentation logits.

### 3.2 Raster Mamba Context Block

Let `E_3` denote the 32 x 32 encoder feature. It is flattened in row-major
order into 1024 tokens:

`X = Flatten_row(E_3), X in R^(1024 x C)`.

The sequence is normalized and processed by an official `mamba-ssm` block:

`Y = Mamba(LayerNorm(X))`.

The output is reshaped and returned through a residual projection:

`E'_3 = E_3 + P(Reshape(Y))`.

The residual path retains local texture information while the state-space path
provides long-range sequence context. Unlike four-directional visual scans,
the main model uses one fixed Raster order, making the placement experiment
reproducible and easy to compare with CNN-only and decoder alternatives.

### 3.3 Boundary-Aware Decoding

Let `S(I)` be the final segmentation logits and `B(I)` the final boundary
prediction. The segmentation objective combines binary cross-entropy and soft
Dice loss:

`L_seg = BCE(S, M) + L_dice(S, M)`.

The boundary head is supervised by:

`L_boundary = BCE(B, G)`,

where `M` is the lesion mask and `G` is the boundary target derived from it.
The complete objective is:

`L = L_seg + 0.4 L_boundary + L_aux`.

`L_aux` contains the fixed-weight auxiliary decoder segmentation losses. The
final configuration does not use encoder multi-scale boundary losses because
the controlled experiments did not show a consistent improvement.

## 4. Experimental Setup

### 4.1 Datasets and Protocol

The experiments use ISIC2017, ISIC2018, and PH2. In the first protocol, the
model is trained on ISIC2018, validated on ISIC2018, and evaluated on ISIC2017
and PH2. In the second protocol, it is trained and validated on ISIC2017, then
evaluated on ISIC2018 and PH2. PH2 is treated as an external test domain. Test
labels are never used for training or model selection.

### 4.2 Training Configuration

All models use 256 x 256 RGB inputs, AdamW with learning rate `1e-3`, weight
decay `1e-4`, batch size 16, at most 300 epochs, and early stopping patience
60. Automatic mixed precision is enabled on the Kaggle T4 GPU. Reported
results use seeds 42, 1234, and 2026. The checkpoint with the best validation
Dice is used for final evaluation.

### 4.3 Metrics

We report Dice, IoU, accuracy, sensitivity, specificity, and HD95. Dice and
IoU measure region overlap. Sensitivity and specificity describe foreground
recall and background rejection. HD95 evaluates contour robustness while
reducing the influence of extreme outlier pixels. Parameters, estimated FLOPs,
model size, and runtime are supporting efficiency measurements, not the main
scientific contribution.

### 4.4 Ablation Models

| Model | Description |
| --- | --- |
| CNN-only | No Mamba; convolutional encoder and final boundary supervision |
| BRSS-Raster | One encoder Raster Mamba at 32 x 32; final boundary supervision |
| Decoder Mamba | Mamba bridge in the decoder without mask conditioning |
| Mask-guided fusion | Coarse-mask fusion without Mamba |
| MGMB | Decoder Mamba bridge conditioned by a predicted coarse mask |
| Uniform-mask MGMB | MGMB with a constant 0.5 mask control |
| MGMB-16 | Complete bridge moved to the 16 x 16 decoder fusion |

The decoder-side models are mechanistic controls. The coarse mask in MGMB is
predicted by the network itself; a ground-truth mask is never supplied at
inference.

## 5. Results

### 5.1 ISIC2018 Training Domain

The following values are means over seeds 42, 1234, and 2026:

| Model | ISIC2018 Dice | ISIC2017 Dice | PH2 Dice | PH2 HD95 |
| --- | ---: | ---: | ---: | ---: |
| CNN-only | 0.8863 | 0.8861 | 0.9177 | 13.46 |
| **BRSS-Raster** | **0.8864** | **0.8896** | 0.9166 | 13.89 |
| Decoder Mamba | 0.8851 | 0.8889 | 0.9162 | 13.62 |
| Mask-guided fusion | 0.8858 | 0.8837 | 0.9110 | 14.53 |
| MGMB | 0.8848 | 0.8855 | 0.9154 | 14.61 |

BRSS-Raster obtains the strongest mean Dice on both ISIC evaluation sets,
although CNN-only remains stronger on PH2. The decoder-side mask-guided
variants do not improve the mean result and show greater seed sensitivity.

### 5.2 ISIC2017 Training Domain

The following values are means over seeds 42, 1234, and 2026:

| Model | ISIC2017 Dice | ISIC2018 Dice | PH2 Dice | PH2 HD95 |
| --- | ---: | ---: | ---: | ---: |
| CNN-only | 0.8744 | 0.8791 | 0.9184 | 13.90 |
| **BRSS-Raster** | 0.8737 | 0.8768 | **0.9217** | 13.78 |
| Decoder Mamba | 0.8755 | 0.8776 | 0.9171 | 13.71 |
| Mask-guided fusion | 0.8739 | 0.8776 | 0.9110 | 13.97 |
| MGMB | **0.8765** | **0.8791** | 0.9165 | 14.26 |

When trained on ISIC2017, BRSS-Raster obtains the strongest PH2 Dice among the
principal CNN-only and encoder-Raster alternatives, and remains competitive
with the decoder-side exploratory controls. Its in-domain score is close to
CNN-only, supporting a competitive cross-domain context module rather than
universal dominance.

### 5.3 Ablation Interpretation

The experiments produce four restrained observations. First, adding Mamba is
not sufficient; scan layout and placement matter. Second, the simple encoder
Raster placement is more stable than the tested mask-guided decoder bridges.
Third, the 16 x 16 bridge can be strong on PH2 for one protocol but is not
consistent across domains. Fourth, the constant-mask control does not establish
a robust advantage for predicted mask conditioning. These negative findings
support focusing the final method on placement and reproducibility rather than
on a more complex mask-conditioned architecture.

## 6. Discussion

The results suggest that one intermediate-resolution Raster Mamba block is a
reasonable compromise between local convolution and global sequence context.
The 32 x 32 representation contains 1024 tokens, allowing long-range modeling
before the decoder restores full resolution. Restricting Mamba to one stage also
limits the number of architectural factors that change simultaneously.

Cross-dataset testing is important because a small validation improvement may
reflect dataset-specific appearance statistics. BRSS-Raster is strongest on PH2
when trained on ISIC2017 and strongest on the ISIC evaluation sets when trained
on ISIC2018. This indicates dataset-dependent benefits rather than universal
dominance. A simple Raster design with transparent controls is therefore more
defensible than a larger collection of unverified attention or mask modules.

The decoder experiments show that a decoder Mamba without mask conditioning
does not consistently improve the baseline, while mask-guided variants show
noticeable seed sensitivity. Adding a semantic mask branch is therefore not
automatically a solution to background interference.

## 7. Limitations

This work does not introduce a new state-space recurrence or scan equation. Its
novelty is architectural placement and controlled validation, which may be
considered incremental by reviewers seeking a new Mamba formulation. The
experiments use fixed 256 x 256 resizing, and FLOPs are estimates based on the
implemented operators. Before submission, the study should include fair
reproduced baselines, batch-1 latency on the same device, paired per-image
statistical tests, qualitative examples, and failure-case analysis.

## 8. Conclusion

We presented BRSS-Raster, a six-stage skin lesion segmenter with one Raster
Mamba block at the 32 x 32 encoder level and final boundary supervision. Across
two training-domain protocols and three random seeds, it remains competitive
with CNN-only segmentation and provides the strongest PH2 Dice among the main
alternatives when trained on ISIC2017. Decoder-side mask-guided Mamba bridges
were not consistently better and were not adopted as the final architecture.
The findings indicate that careful Mamba placement and cross-dataset evaluation
are more reliable than simply increasing the number of state-space modules.

## 9. Submission Checklist

1. Add fair results for U-Net, UNet++, Attention U-Net, EGE-UNet, VM-UNet, and
   recent lightweight Mamba models using the same split and image protocol.
2. Measure inference latency and FPS with batch size 1 after warm-up on a fixed
   T4 device.
3. Run paired per-image Dice tests and confidence intervals for BRSS-Raster
   versus CNN-only and the strongest baseline.
4. Add qualitative figures for low contrast, hair artifacts, irregular edges,
   and small lesions, including representative failures.
5. Recompute all tables from archived run folders and verify that every mean and
   standard deviation uses the same three seeds.
6. Verify that the exact implementation and all split definitions are publicly
   reproducible.
