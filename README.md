# Deep Spatial Omics: Metabolic Gradient Modeling

This repository contains the implementation and expansion of Deep Learning architectures (GASTON and MET-MAP) for modeling and predicting continuous spatial gradients in spatial metabolomics data.

## Project Overview

The objective of this project is to leverage neural networks, specifically Multi-Layer Perceptrons (MLPs) implemented in PyTorch, to predict metabolite expression based strictly on geographical tissue coordinates (x, y).

We apply MET-MAP to the BC-515 breast cancer section: 3,508 spots, 2,086 metabolites.

## Getting started

See [SETUP.md](SETUP.md) to create the environment, download the data and run the pipeline.

## Project layout

```text
config.yaml                          default parameters (original MET-MAP settings) and the models compared with --cv
src/
├── main.py                          runs the full pipeline (one run, or the evaluation with --cv)
├── data.py                          load BC-515, TIC normalization, standardization
├── model.py                         model classes behind a shared contract, and MODEL_REGISTRY
│                                    (MetMapMLP: encoder (x, y) -> isodepths, linear decoder -> metabolites)
├── train.py                         training loop shared by all models (Adam, or AdamW if weight_decay > 0)
├── evaluate.py                      per-metabolite R2, scaled loadings, ranking table
├── splits.py                        random 80/20 repeats, spatial k-fold, Tumor/Stroma/Interface coverage
├── cv.py                            fresh model per split, held-out metrics per split and per metabolite
├── viz.py                           figures, including the model-comparison boxplot
└── dummy_data.py                    small synthetic dataset for running without the real file
analysis/
├── tumor_stroma.py                  biology of one run: Tumor vs Stroma changes, model R2, likely
│                                    identities (paper or HMDB) and their profile along the isodepth
├── hmdb_candidates.py               HTML report with all HMDB candidates per metabolite
└── build_hmdb_index.py              builds data/hmdb_compact.csv from the full HMDB XML (run once)
data/
├── BC_515_Section_1.h5ad            the dataset, download it (not tracked by git)
├── hmdb_compact.csv                 reduced HMDB 5.0 table used for the mass search
└── annotation_reference_table1.csv  annotations of the original paper
results/                             one folder per experiment (not tracked by git)
```
 
## Adding a model

Only `src/model.py` and `config.yaml` change. See the instructions at the top of `src/model.py`.

## Contributions

**EduSR9 contribution:**

- **Project structure:** I set up the `src/` pipeline, `config.yaml`, the environment files and
  the documentation (`README.md`, `SETUP.md`), and wrote `src/main.py`, the script that runs
  either one training run or the full evaluation.
- **Modular models:** I designed the shared contract and `MODEL_REGISTRY` in `src/model.py`, and
  the optimizer choice in `src/train.py` (Adam, or AdamW with weight decay). This way, anyone can
  add a new model with a few lines and compare it with MET-MAP without touching the rest of the code.
- **Evaluation:** fitting all spots only shows how well the model reconstructs what it has seen,
  so I added tests on spots it has never seen. `src/splits.py` makes random 80/20 splits and
  spatial folds (whole regions of tissue held out), and checks that Tumor, Stroma and the
  Interface are represented in every fold. `src/cv.py` trains a new model for each split, and
  `src/evaluate.py` scores every metabolite with R2. The boxplot in `src/viz.py`
  (`plot_model_boxplots`) puts reconstruction, random and spatial side by side for every model.
- **Biology (`analysis/`):** I looked for the metabolites that differ most between Tumor and
  Stroma and checked which of them the model explains well. To know what those peaks are, I matched
  them with the annotations of the original paper of the dataset we used and 
  with a reduced copy of the HMDB 5.0 database.
  Finally, I plotted the well explained metabolites along the isodepth that follows the tumor.

*Use of GenAI:* I used Claude (Anthropic) as a coding assistant for writing and debugging code,
comments and the wording of this README. Prompts were mainly requests to fix errors, make the code
modular, adapt it to new needs and add comments, for example: "help me make the model.py code modular so my
teammates can plug in their models", "the boxplot breaks when near-constant metabolites give R2
values of -1e15, help me fix it", and "make the biology analysis work for datasets that don't have exactly 2,086
metabolites". Over several weeks I decided what each part had to do, ran every step on the real
data and checked the results before keeping them.

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
