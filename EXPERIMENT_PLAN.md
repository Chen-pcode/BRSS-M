# BRSS-MambaSeg Experiment Plan

## Hypothesis

At low resolution, a selective state-space block can capture lesion-scale
context without high-resolution attention cost. Its global propagation should
be conditioned by boundary uncertainty so that ambiguous contours retain local
evidence instead of spreading artifacts such as hair and illumination changes.

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
| 4-stage, parameter-matched | Remove deep 16/8 levels while increasing width | Value of six-resolution hierarchy |
| 5-stage | Intermediate depth control | Depth-response curve |
| w/o selective SSM | CNN only at deep stages | Value of state-space dynamics |
| Plain cumulative scan | Replace SSM with fixed scan | Dynamic SSM vs BLMNet-style aggregation |
| No boundary router | Remove routing from SSM and decoder | Overall routing contribution |
| SSM router only | Keep routing only during state-space propagation | State-update contribution |
| Decoder router only | Keep routing only during skip fusion | Decoder-only control |
| w/o local path | Remove depthwise local feature path | Local-global complementarity |
| w/o cross-scale fusion | Remove global skip context | Cross-scale fusion contribution |
| Final boundary supervision only | Remove deep boundary supervision | Multi-scale structural supervision |
| w/o boundary loss | Keep architecture, remove all boundary loss | Objective-level contribution |

Do not claim a component improves performance unless its three-seed mean and
paired per-image Dice comparison are consistent. For the routing claim, report
both segmentation metrics and the Dice/IoU of each supervised boundary scale.
All models must use the same split manifests and protocol-specific checkpoint
selection.
