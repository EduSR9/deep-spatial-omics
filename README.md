# Deep Spatial Omics: Metabolic Gradient Modeling

This repository contains the implementation and expansion of Deep Learning architectures (GASTON and MET-MAP) for modeling and predicting continuous spatial gradients in spatial metabolomics data.

## Project Overview

The objective of this project is to leverage neural networks, specifically Multi-Layer Perceptrons (MLPs) implemented in PyTorch, to predict metabolite expression based strictly on geographical tissue coordinates (x, y).

We apply MET-MAP to the BC-515 breast cancer section: 3,508 spots, 2,086 metabolites.

## Getting started

See [SETUP.md](SETUP.md) to create the environment, download the data and run the pipeline.

## Project layout

```text
config.yaml          default parameters (original MET-MAP settings) and the list of
                     models compared with --cv ("models:")
src/
  main.py            runs the full pipeline
  data.py            load BC-515, TIC normalization, standardization
  model.py           model classes behind a shared contract, and MODEL_REGISTRY
                     (MetMapMLP: encoder (x, y) -> isodepths, linear decoder -> metabolites)
  train.py           training loop shared by all models: MSE plus the model's own penalty;
                     Adam, or AdamW when weight_decay > 0
  evaluate.py        per-metabolite R^2, scaled loadings, ranking table
  splits.py          random 80/20 repeats, spatial k-fold, Tumor/Stroma/Interface coverage
  cv.py              fresh model per split, held-out metrics per split and per metabolite
  viz.py             figures, including the model-comparison boxplot
  dummy_data.py      small synthetic dataset for running without the real file
analysis/
  tumor_stroma.py    biology of one run: Tumor vs Stroma changes, model R2, likely
                     identities (paper or HMDB) and their profile along the isodepth
  hmdb_candidates.py detailed HTML report with all HMDB candidates per metabolite
  build_hmdb_index.py  builds data/hmdb_compact.csv from the full HMDB XML (run once)
data/                put BC_515_Section_1.h5ad here (not tracked by git)
  hmdb_compact.csv   reduced HMDB 5.0 table used for the mass search (tracked)
  annotation_reference_table1.csv  annotations of the original paper (tracked)
results/             one folder per experiment (not tracked by git)
```

Open work is marked with `# TODO` in each file.

## Adding a model

Only `src/model.py` and `config.yaml` change. See the instructions at the top of `src/model.py`.

## Reference

Our implementation follows notebook `03_METMAP_from_scratch.ipynb` in
[Elsa-bingxue/METMAP-SIGMA-Tutorials](https://github.com/Elsa-bingxue/METMAP-SIGMA-Tutorials).
With the default `config.yaml`, our results should match that notebook.

- MET-MAP: [raphael-group/MET-MAP](https://github.com/raphael-group/MET-MAP)
- Dataset and metabolite annotations (`data/annotation_reference_table1.csv`, Supporting
  Table 1): Godfrey TM et al. Integrating Ambient Ionization Mass Spectrometry Imaging and
  Spatial Transcriptomics on the Same Cancer Tissues to Identify RNA-Metabolite Correlations.
  *Angew. Chem. Int. Ed.* 2025, 64, e202502028. https://doi.org/10.1002/anie.202502028
- HMDB 5.0 (`data/hmdb_compact.csv`): Wishart DS et al. HMDB 5.0: the Human Metabolome
  Database for 2022. *Nucleic Acids Res.* 2022, 50(D1), D622-D631.
  https://doi.org/10.1093/nar/gkab1062
