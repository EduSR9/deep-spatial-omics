# Deep Spatial Omics: Metabolic Gradient Modeling

This repository contains the implementation and expansion of Deep Learning architectures (GASTON and MET-MAP) for modeling and predicting continuous spatial gradients in spatial metabolomics data.

## Project Overview

The objective of this project is to leverage neural networks, specifically Multi-Layer Perceptrons (MLPs) implemented in PyTorch, to predict metabolite expression based strictly on geographical tissue coordinates (x, y).

We apply MET-MAP to the BC-515 breast cancer section: 3,508 spots, 2,086 metabolites.

## Getting started

See [SETUP.md](SETUP.md) to create the environment, download the data and run the pipeline.

## Project layout

```text
config.yaml          default parameters (original MET-MAP settings)
src/
  main.py            runs the full pipeline
  data.py            load BC-515, TIC normalization, standardization
  model.py           MetMapMLP: encoder (x, y) -> isodepths, linear decoder -> metabolites
  train.py           loss (MSE + L1 on decoder weights) and training loop
  evaluate.py        per-metabolite R^2, scaled loadings, ranking table
  viz.py             figures
  dummy_data.py      small synthetic dataset for running without the real file
data/                put BC_515_Section_1.h5ad here (not tracked by git)
results/             one folder per experiment (not tracked by git)
```

Open work is marked with `# TODO` in each file.

## Reference

Our implementation follows notebook `03_METMAP_from_scratch.ipynb` in
[Elsa-bingxue/METMAP-SIGMA-Tutorials](https://github.com/Elsa-bingxue/METMAP-SIGMA-Tutorials).
With the default `config.yaml`, our results should match that notebook.

- MET-MAP: [raphael-group/MET-MAP](https://github.com/raphael-group/MET-MAP)
