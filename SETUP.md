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
```

Every run writes to `results/<exp-name>/`, including the exact `config.yaml` used.

Available overrides: `--n-features`, `--n-isodepth`, `--hidden-dim`, `--epochs`,
`--learning-rate`, `--lasso-lambda`, `--seed`.
