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
python src/main.py --exp-name iso5 --n-isodepth 5           # using 5 isodepths
python src/main.py --exp-name metmap_run --model metmap     # one model, with its settings from "models:"
python src/main.py --exp-name cv --cv                       # evaluate every model in "models:"
python src/main.py --exp-name gat_cv --cv --model attention # evaluate graph attention only
```

Single-model runs write to `results/<model>/<exp-name>/`, with `baseline` as the default experiment name.
An all-model CV run writes to `results/<exp-name>/`. Each model's folder contains the effective `config.yaml`.

### Single run (no `--cv`)

Trains one model on all spots and writes the isodepth maps, training loss, top
metabolites and the metabolite ranking. Without `--model`, it uses the `model:` and
`training:` sections of `config.yaml`. 

There are multiple ways to run the proposed model extensions: 
- by changing the `model:` inside the `config.yaml` to another
- via the command-line flag `--model <name>` which overrides values specified in `config.yaml`
Supported model names are `metmap`, `attention` (64-neighbor graph attention),
`self_attention` (metabolite-token self-attention), `xym`, `joint`, and `pca`.

### Cross-validation (`--cv`)

Without `--model`, evaluates every model in `models:` on shared random splits and spatial folds.
By default (`evaluation.n_features: null`) the all-model comparison uses all metabolites. Set
`evaluation.n_features` to a number, or pass `--n-features`, to keep only the top-n per split,
selected using the training spots. The `self_attention` decoder attends across metabolites
and is intended for small feature subsets. With `--model`, the same protocol runs for the selected
model and writes to `results/<model>/<exp-name>/`.

- `coverage.csv`, `spatial_folds.png`: region counts per spatial fold, shared by all models.
- `cv_results.csv`, `cv_summary.csv`: held-out metrics per split, and their mean and spread.
- `cv_feature_r2.csv`: held-out R² for each metabolite and split.
- `reconstruction_r2.csv`: in-sample R² per metabolite, from a separate fit on all spots. It is
  not part of the held-out evaluation.
- `boxplot_r2.png`, `boxplot_summary.csv`: per-metabolite R² averaged over splits, with one box
  per model and protocol. The figure has a panel for models that use coordinates alone and a
  panel for XYM, Joint, and PCA, which use measured metabolites at the test spots.

Spatial CV is the main evaluation; random splits are a secondary check. Settings for the
splits (number of repeats, number of folds, interface definition) are in the
`evaluation` section of `config.yaml`.

#### Quick test (`config_fast.yaml`)

`config.yaml` is the default and the one to use for real results; every command uses it unless
`--config` says otherwise. `config_fast.yaml` is only a quick-test copy of it, to check that the
pipeline runs: 1 random split, 2 spatial folds and 10 metabolites. With `--dummy` and few epochs,
an all-model CV finishes quickly:

```bash
python src/main.py --dummy --cv --config config_fast.yaml --epochs 10 --exp-name fast_check_cv
```

Drop `--dummy` to run the same quick check on the real data. If you change the models in
`config.yaml`, update `config_fast.yaml` too.

With a CUDA-enabled PyTorch installation, set `device: cuda` under `training` in a copy of
`config.yaml`. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the shell before running Python for
deterministic CUDA operations.

Available overrides: `--model`, `--data`, `--n-features`, `--n-isodepth`, `--hidden-dim`, `--n-pcs`,
`--epochs`, `--learning-rate`, `--lasso-lambda`, `--weight-decay`, `--seed`.

### Adding a model

A new model can be added by following the steps below:
1. Create a new model file in `src` folder and, if needed, a corresponding training file.
2. Register the model in `src/factory.py`.
3. Add any model-specific parameters to `config.yaml`
4. Add CLI overrides in `src/main.py` if those parameters should be configurable from the command
   line
5. Ensure the model follows the project's input/output conventions so it can use existing training,
   evaluation, plotting, and cross-validation

Test new models with:
```bash
python src/main.py --dummy --exp-name test
python src/main.py --dummy --cv --epochs 2 --exp-name test_cv
```

## 4. Biology analysis

Use the outputs from a single run on the real data, for example `results/metmap/baseline/`.
The HMDB table and the paper annotations are already in `data/`, so nothing else has to be
downloaded.

```bash
python analysis/tumor_stroma.py metmap/baseline
python analysis/hmdb_candidates.py metmap/baseline
```

Both write to `results/biology/<model>/<exp-name>/`:

- `change_vs_r2.png`: every metabolite, Tumor - Stroma difference vs model R2.h
- `well_explained_identities.png`: the big changes the model explains well (R2 > 0.6), wit
  their likely identity and chemical class.
- `isodepth_profiles.png`: those metabolites along the isodepth that best follows Tumor/Stroma.
- `big_change_identities.csv`, `biology_summary.csv`, `tumor_stroma_table.csv`.
- `hmdb_report.html`, `hmdb_candidates.csv`: all HMDB candidates for each big change.

Names without a mark come from the paper; `?` is an HMDB mass match within 10 ppm and `??`
only within 50 ppm. They are mass matches, not identifications. Runs trained on fewer
metabolites (`n_features`) also work: metabolites the model did not use have no R2.

`data/hmdb_compact.csv` only has to be rebuilt for a new HMDB version: download
`hmdb_metabolites.xml` (about 6.5 GB) from hmdb.ca into `data/` and run
`python analysis/build_hmdb_index.py`.
