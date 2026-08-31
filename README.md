# BRSS-MambaSeg

BRSS-MambaSeg is an isolated, Kaggle-ready research project for skin lesion
segmentation. It evaluates a six-resolution CNN-selective-state-space network
with boundary-uncertainty-routed decoder fusion.

The state-space block is a portable PyTorch reference implementation with
input-dependent delta, B and C parameters. It is not described as VMamba or as
an official Mamba kernel. This distinction is deliberate and should be retained
in any paper.

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
   listed packages only if the Kaggle image does not already provide them.
3. Run a smoke test before a full experiment:

```bash
python train.py --epochs 2 --workers 2 --amp --output-dir /kaggle/working/smoke
```

4. Run one proposed-model seed, inspect `config.json`, `history.csv`, and
   `summary.csv`, then run the controlled suite:

```bash
python run_ablations.py --amp --output-root /kaggle/working/ablation --skip-completed
```

The core suite is 36 training jobs for ISIC2018 (12 variants x 3 seeds). Run
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
