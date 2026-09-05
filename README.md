# BRSS-MambaSeg

BRSS-MambaSeg is an isolated, Kaggle-ready research project for skin lesion
segmentation. It uses official `mamba-ssm` Mamba blocks to build global context
at low-resolution encoder features, with a local convolutional path and
multi-scale boundary supervision.

`AxialMamba2D` runs four official Mamba blocks over row-major and column-major
tokens in both directions. It is applied only at 16 x 16 and 8 x 8 features
for 256 x 256 inputs. The model is therefore Mamba-based, but it is not a
VMamba/SS2D reproduction. The prior boundary routers and cross-scale mean
injection were removed because their ablations did not show consistent gains.

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
python train.py --epochs 2 --batch-size 16 --workers 2 --amp --output-dir /kaggle/working/smoke
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

The core suite is 21 training jobs for ISIC2018 (7 variants x 3 seeds). Run
the full suite only after the smoke test and one single-seed full-model run.
Repeat it on ISIC2017 with:

```bash
python run_ablations.py --train-dataset isic2017 --val-dataset isic2017 --test-datasets isic2018 PH2 --amp --output-root /kaggle/working/ablation_isic2017
```

## Data Layout

```
/kaggle/input/datasets/zichengdoctor/isic2017/{train,val}/{images,masks}
/kaggle/input/datasets/zichengdoctor/isic2018/{train,val}/{images,masks}
/kaggle/input/datasets/zichengdoctor/ph2dataset/ph2/test/{images,masks}
```

## Output Contract

Each run writes a checkpoint, immutable runtime/configuration metadata, epoch
history, per-image CSV files and a protocol-level `summary.csv`. `summarize.py`
creates `all_runs.csv` and `ablation_mean_std.csv`; these are the only files to
use for paper tables.

## Interrupted Runs

`latest.pt` is saved after every completed epoch. After attaching a prior
Kaggle output dataset and restoring its run directory, continue with:

```bash
python train.py --resume --output-dir /kaggle/working/full_seed2026 --amp
```
