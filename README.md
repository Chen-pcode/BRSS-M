# BRSS-MambaSeg

BRSS-MambaSeg is an isolated, Kaggle-ready research project for skin lesion
segmentation. It uses official `mamba-ssm` Mamba blocks to build global context
at low-resolution encoder features, with multi-scale boundary supervision.

`HighResolutionGroupedMamba` applies one shared official Mamba block to
row-major and column-major sequences at the 32 x 32 feature level. Channels
are compressed and split into two groups before the Mamba scan, with groups
folded into the batch dimension to share all Mamba parameters. The model is
therefore Mamba-based, but it is not a VMamba/SS2D reproduction.

The current candidate direction is a coarse-mask-conditioned Mamba bridge.
It uses a predicted 16 x 16 decoder mask to split the following 32 x 32
cross-scale fusion into soft lesion and background streams, scans both streams
with one shared Raster Mamba, and restores them through a residual fusion.
No ground-truth mask enters this bridge at inference or training.

## Layout

```
BRSS-MambaSeg/
  brss/                model, data, loss, metrics and training engine
  train.py             one model / one seed / all evaluation protocols
  run_ablations.py     controlled multi-seed ablation suite
  summarize.py         raw and mean-plus-standard-deviation results
  EXPERIMENT_PLAN.md   preregistered experimental claims
```

## Kaggle Setup

1. Create a Kaggle Dataset containing this directory, then attach it to a GPU
   notebook. The default Kaggle roots are already configured as
   `/kaggle/input/datasets/zichengdoctor/isic2017`,
   `/kaggle/input/datasets/zichengdoctor/isic2018`, and
   `/kaggle/input/datasets/zichengdoctor/ph2dataset`.
2. Set the working directory to the uploaded code directory and install the
   dependencies. `mamba-ssm` needs a CUDA-compatible binary or compiler:

```bash
pip install --no-build-isolation mamba-ssm causal-conv1d
```
3. Run a smoke test before a full experiment:

```bash
python train.py --model brss_hgm_mamba --epochs 2 --batch-size 16 --workers 2 --amp --output-dir /kaggle/working/smoke
```

Loss-function ablations can be run directly with their experiment names. They
use the full BRSS-Mamba architecture while disabling the indicated supervision:

```bash
python train.py --model brss_no_boundary_loss --amp --output-dir /kaggle/working/no_boundary_loss
python train.py --model brss_final_boundary_only --amp --output-dir /kaggle/working/final_boundary_only
```

4. Run one proposed-model seed, inspect `config.json`, `history.csv`, and
   `summary.csv`, then run the controlled suite:

```bash
python run_ablations.py --amp --output-root /kaggle/working/ablation --skip-completed
```

The HGM suite is 33 training jobs for ISIC2018 (11 variants x 3 seeds): four
Mamba-placement variants (32 x 32; 32 x 32 + 16 x 16; 32 x 32 + 16 x 16 + 8 x
8; and 16 x 16 only), plain Raster Mamba, no-Mamba, no-compression,
no-grouping, single-axis, no-boundary-loss and final-boundary-only variants. Run
the full suite only after the smoke test and one single-seed full-model run.
Repeat it on ISIC2017 with:

```bash
python run_ablations.py --train-dataset isic2017 --val-dataset isic2017 --test-datasets isic2018 PH2 --amp --output-root /kaggle/working/ablation_isic2017
```

Run the decoder-bridge study separately. Its five variants all use final
boundary supervision only: CNN-only, encoder Raster Mamba, decoder Mamba
without a mask, mask-guided fusion without Mamba, and the full bridge.

```bash
python run_ablations.py --models brss_cnn_final_boundary brss_raster_final_boundary brss_decoder_mamba_bridge brss_mask_guided_fusion brss_mgmb_mamba_bridge --seeds 2026 --amp --output-root /kaggle/working/mgmb_ablation
```

## Data Layout

```
/kaggle/input/datasets/zichengdoctor/isic2017/{train,val}/{images,masks}
/kaggle/input/datasets/zichengdoctor/isic2018/{train,val}/{images,masks}
/kaggle/input/datasets/zichengdoctor/ph2dataset/ph2/test/{images,masks}
```

## Output Contract

Each run writes a checkpoint, immutable runtime/configuration metadata, epoch
history, per-image CSV files and a protocol-level `summary.csv`. The summary
reports Params (M), estimated FLOPs (G) for one 256 x 256 image, FP32 model
size (MB), Dice, IoU, accuracy, sensitivity, specificity and HD95.
`summarize.py` creates `all_runs.csv` and `ablation_mean_std.csv`; these are
the only files to use for paper tables.

## Interrupted Runs

`latest.pt` is saved after every completed epoch. After attaching a prior
Kaggle output dataset and restoring its run directory, continue with:

```bash
python train.py --resume --output-dir /kaggle/working/full_seed2026 --amp
```
