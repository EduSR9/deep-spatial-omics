# Setup

All commands run from the project root (`deep-spatial-omics/`).

## 1. Environment

```bash
conda env create -f environment.yml
conda activate dsomics
```

To rebuild after `requirements.txt` changes:

```bash
conda env remove -n dsomics
conda env create -f environment.yml
```

## 2. Data

Download [BC_515_Section_1.h5ad](https://github.com/Elsa-bingxue/METMAP-SIGMA-Tutorials/releases/download/bc515-data-v1/BC_515_Section_1.h5ad) (about 108 MB) and place it at:

```text
data/BC_515_Section_1.h5ad
```

## 3. Run

```bash
python src/main.py --dummy                                  # synthetic data, no file needed
python src/main.py --exp-name baseline                      # real data, config.yaml defaults
python src/main.py --exp-name top100 --n-features 100       # only the 100 most abundant metabolites
python src/main.py --exp-name iso5 --n-isodepth 5
python src/main.py --exp-name metmap_run --model metmap     # one model, with its settings from "models:"
python src/main.py --exp-name cv --cv                       # evaluation: every model in "models:"
```

Every run writes to `results/<exp-name>/`, including the exact `config.yaml` used.

### Single run (no `--cv`)

Trains one model on all spots and writes the isodepth maps, training loss, top
metabolites and the metabolite ranking. Without `--model`, it uses the `model:` and
`training:` sections of `config.yaml`. With `--model <name>`, it also applies that
model's `params` and `training` overrides from the `models:` list. Command-line flags
override everything else.

### Cross-validation (`--cv`)

Runs every model in the `models:` list of `config.yaml`, on the same splits, and writes:

- `coverage.csv`, `spatial_folds.png`: region counts per spatial fold, shared by all models.
- `<model name>/cv_results.csv`, `<model name>/cv_summary.csv`: held-out metrics per split,
  and their mean and spread.
- `<model name>/cv_feature_r2.csv`: held-out R² for each metabolite and split.
- `<model name>/reconstruction_r2.csv`: in-sample R² per metabolite, from a separate fit on
  all spots. It is not part of the held-out evaluation.
- `boxplot_r2.png`, `boxplot_summary.csv`: one box per model for reconstruction (in-sample),
  random and spatial; each box holds one value per metabolite, averaged over splits.

Spatial CV is the main evaluation; random splits are a secondary check. Settings for the
splits (number of repeats, number of folds, interface definition) are in the
`evaluation` section of `config.yaml`. For a quick test, copy `config.yaml`, lower
`n_random_repeats` and `n_spatial_folds` (at least 2), and pass it with `--config`.

Available overrides: `--model`, `--data`, `--n-features`, `--n-isodepth`, `--hidden-dim`,
`--epochs`, `--learning-rate`, `--lasso-lambda`, `--weight-decay`, `--seed`.

### Adding a model

See "HOW TO ADD A NEW MODEL" at the top of `src/model.py`, and the comments above
`models:` in `config.yaml`. Test a new model with:

```bash
python src/main.py --dummy --model my_model --exp-name test_my_model
python src/main.py --dummy --cv --exp-name test_cv
```
