# Deep Spatial Omics: Metabolic Gradient Modeling

This repository contains the implementation and expansion of Deep Learning architectures (GASTON and MET-MAP) for modeling and predicting continuous spatial gradients in spatial metabolomics data.

## Project Overview

The objective of this project is to leverage neural networks, specifically Multi-Layer Perceptrons (MLPs) implemented in PyTorch, to predict metabolite expression based strictly on geographical tissue coordinates (x, y).

We apply MET-MAP to the BC-515 breast cancer section: 3,508 spots, 2,086 metabolites.

## Getting started

See [SETUP.md](SETUP.md) to create the environment, download the data and run the pipeline.

## Project layout

```text
config.yaml                          default parameters (original MET-MAP settings), model selection,
                                     training parameters, PCA settings, and the models compared with --cv
src/
├── main.py                          runs the full pipeline (one run, or the evaluation with --cv)
├── factory.py                       model creation, training dispatch, prediction dispatch,
│                                    and PCA preprocessing behind a shared interface
├── data.py                          load BC-515, TIC normalization, feature selection,
│                                    standardization, and train/test preprocessing
├── model.py                         original MET-MAP model
│                                    (MetMapMLP: encoder (x, y) -> isodepths, linear decoder -> metabolites)
├── attention.py                     attention-based MET-MAP variant
├── model_xym.py                     XYM model: encoder (x, y, metabolites) -> isodepths,
│                                    linear decoder -> metabolites
├── model_joint.py                   Joint Spatial-Metabolic model: encoder (x, y, metabolites) -> isodepths;
│                                    decoder 1: isodepth -> (x, y), decoder 2: isodepth -> metabolites
├── model_pca.py                     PCA-enhanced MET-MAP model:
│                                    encoder (x, y, PCs) -> isodepths
├── train.py                         training loop for MET-MAP and Attention models
│                                    (Adam, or AdamW if weight_decay > 0)
├── train_xym.py                     training loop for the XYM model
├── train_joint.py                   training loop for the Joint Spatial-Metabolic model
├── train_pca.py                     training loop for the PCA model
├── evaluate.py                      per-metabolite R², scaled loadings, ranking table
├── splits.py                        random 80/20 repeats, spatial k-fold, Tumor/Stroma/Interface coverage
├── cv.py                            fresh model per split, held-out metrics per split and per metabolite
├── viz.py                           figures for annotations, training loss, isodepths,
│                                    top metabolites, gradient profiles, and CV visualizations
└── dummy_data.py                    small synthetic dataset for running without the real file
analysis/
├── tumor_stroma.py                  biology of one run: Tumor vs Stroma changes, model R², likely
│                                    identities (paper or HMDB) and their profile along the isodepth
├── hmdb_candidates.py               HTML report with all HMDB candidates per metabolite
└── build_hmdb_index.py              builds data/hmdb_compact.csv from the full HMDB XML (run once)
data/
├── BC_515_Section_1.h5ad            the dataset, download it (not tracked by git)
├── hmdb_compact.csv                 reduced HMDB 5.0 table used for the mass search
└── annotation_reference_table1.csv  annotations of the original paper
results/                             one folder per model and experiment
                                     (not tracked by git)
```

## Adding a model

Files for model setup and training are created. `src/factory.py`, `config.yaml` are to be updated. 
See more detailed instructions at the top of `src/model.py`.

## Contributions

**Cheng Yihan’s code contributions**

- **MET-MAP model (`src/model.py`)**. I implemented the coordinate encoder, linear decoder, and forward pass based on the course materials. I adapted the model to the project’s shared interface.
- **Training code (`src/train.py`)**. I implemented the MET-MAP training loop with mean squared error, an L1 penalty on the decoder, Adam, and learning-rate scheduling. I later integrated the attention model into the shared training loop and added GPU support.
- **Graph attention extension (`src/model.py`, `config.yaml`)**. I added a decoder that selects 64 training neighbors in three-dimensional isodepth space and learns how much each neighbor contributes to the prediction. I compared it with MET-MAP on the same random splits and spatial folds of BC-515 and tested different settings.
- **Other contributions**. I fixed a preprocessing leak in cross-validation and helped with per-metabolite R² exports and the biological analysis workflow.

*GenAI use for coding and language editing.* I used OpenAI Codex to help write and debug parts of the PyTorch code and to run and check model comparisons. I also used it to polish the English wording of this contribution statement. I asked Codex to integrate the graph attention model into the existing code structure, keep the implementation concise, and shorten the descriptions of my secondary contributions.


**Eduardo Iglesias's contributions:**

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
modular, adapt it to new needs and add comments, for example: "help me make the model.py code modular 
so my teammates can plug in their models", "the boxplot breaks when near-constant metabolites give R2
values of -1e15, help me fix it", and "make the biology analysis work for datasets that don't have 
exactly 2,086 metabolites". Over several weeks I decided what each part had to do, ran every step on 
the real data and checked the results before keeping them.

**Fokina Yelizaveta's contributions:**

- **Data preprocessing (`src/data.py`)**: implemented BC-515 data loading and preprocessing for future
  use by all models. Includes extraction of specific features and performing quality control. Data
  preprocessing contains total-ion-count (TIC) normalization, log transformation, standardization.
- **Factory-based model integration (`src/factory.py`)**: implemented a central factory module
  that acts as a dispatcher for all model-related operations. This allows to remove the model-specific
  logic from the main pipeline, making the integration of all models introduced easier.
- **Model extensions**: Introduced and tested 3 alternative models on the breast cancer data:
  1. *Coordinates and metabolites encoder, 1 decoder*: an autoencoder model that uses both coordinates
     and metabolite data at the encoder stage; the decoder used is same as in MET-MAP. Produced
     `src/model_xym.py` and `src/train_xym.py` files, which include training, evaluation, progress
     logging, L1 regularisation, and cross-validation support.
  3. *Coordinates and metabolites encoder, 2 decoders*: similar to previous model, this one uses
     both coordinates and metabolites in the encoder. Modified is the usage of 2 decoders: one for
     coordinates, and one for metabolites. Produced `src/model_joint.py` and `src/train_joint.py`
     files.
  5. *Coordinates and 3 PCs (from metabolites) encoder, 1 decoder*: similarly to model 1, it uses 1
     decoder; for the encoder, coordinates and first 3 principal components of metabolites are used.
     Produced `src/model_pca.py` and `src/train_pca.py` files.
- **Visualisations**: Developed the visualisation module used to explore, evaluate, and compare model
  outputs. Includes tissue annotation maps, spatial cross-validation folds, training loss curves,learned
  isodepth visualizations, top-ranked metabolite reconstructions, and metabolite gradient profiles.

*Use of GenAI*: I have used Copilot as coding assistant for writing and troubleshooting the code, writing 
comments and refining the parts of README and SETUP. Some prompts used are: "Write a Python function to 
map the isodepths side by side from (x, y) coordinates, scaling each isodepth between 0 and 1, and save 
the plot as an image" for writing the function on visualization; it was then verified by me to use the 
correct input data and dimensions. Another is "After running, my code output is always saved to 
`results/default` folder. Help me modify this part so that a folder named after the model type used will 
be created, with the `baseline` folder set as default and other folders created if adjustments are made", 
with the corresponding code attached after. Example for troubleshooting: "My code output (lines with
model learning progress) are displayed in a strange manner. Please look at this piece of code and help me 
identify whether it might be the cause, and how can it be fixed", along with the output and code. I have 
analyzed and double checked the generated code outputs to ensure it was correct.

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
