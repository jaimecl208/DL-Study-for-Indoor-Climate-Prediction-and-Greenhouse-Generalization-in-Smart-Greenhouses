# Deep Learning Study for Indoor Climate Prediction and Cross-Greenhouse Transfer in Smart Greenhouses

Research data and code accompanying the paper:

> J. Carlavilla-Lázaro, V. C. Sánchez, C. T. Calafate, J. M. Cecilia.
> *A Comparative Deep Learning Study for Indoor Climate Prediction and Cross-Greenhouse Transfer in Smart Greenhouses.*
> Submitted to *Computers and Electronics in Agriculture*.

The repository contains the processed datasets, the preprocessing and experiment scripts, the raw prediction outputs, the per-run metrics and the notebooks used to build the tables and figures of the paper.

## Overview

The study forecasts **indoor dry-bulb temperature** and **indoor relative humidity** 6 hours ahead (24 steps at 15-minute resolution) in two commercially operated greenhouses in Spain with contrasting climates:

| Installation | Climate (Köppen-Geiger) | Crop | Modeled unit | Records (processed, 15 min) |
|---|---|---|---|---|
| Álava (northern Spain) | Temperate (C) | Potato | Single compartment | 78,336 |
| Cartagena (southeastern Spain) | Dry / semi-arid (B) | Canary tomato | Compartment C0 | 68,636 |

Three experiments are included:

1. **Architecture comparison.** Seven architectures from four families, trained under the same splits, metrics and protocol at each installation: LSTM, GRU, CNN-LSTM (recurrent / convolutional-recurrent), NLinear, DLinear (linear) and PatchTST, iTransformer (Transformer-based).
2. **Feature-importance analysis.** SHAP values on an auxiliary random-forest model, used to identify actuator variables (window and thermal-screen positions) that act as proxies of the predicted variable.
3. **Cross-greenhouse transfer.** iTransformer pretrained on one greenhouse and evaluated on the other in both directions (Álava → Cartagena, Cartagena → Álava) under three conditions: direct transfer (zero-shot), fine-tuning with increasing amounts of target data, and a target-only reference model. Only the variables common to both installations are used.

### Main results

| | Álava | Cartagena C0 |
|---|---|---|
| Best architecture (temperature) | iTransformer, R² = 0.889 | iTransformer, R² = 0.934 |
| Best architecture (relative humidity) | iTransformer, R² = 0.829 | PatchTST (MSE) / iTransformer (MAE, MAPE) |

Transfer was asymmetric: fine-tuning a model pretrained on Álava with about 200–250 days of Cartagena data matched a Cartagena-only model, whereas in the opposite direction fine-tuning closed 48% (temperature) and 91% (relative humidity) of the R² gap to an Álava-only model. See the paper for the full tables and discussion.

## Repository structure

```
.
├── 00_run_scripts.py              # Runs every experiment script N times and logs the output
├── requirements.txt
├── data/
│   └── processed/                 # Datasets used by the experiments (see "Data")
├── scripts/
│   ├── 01_preprocessing/          # Raw data -> processed datasets, and SHAP feature-importance analysis
│   └── 02_experiments/
│       ├── alava/                 # 01–10: one script per architecture and target
│       ├── cartagena/             # Same set of scripts for Cartagena C0
│       └── finetuning/            # Cross-greenhouse transfer with iTransformer
├── results/
│   ├── experiments/<site>/<model>/<target>/   # Metrics and configuration of every run (JSON)
│   ├── predictions/<site>/<model>/<target>/   # Test-set predictions of every run (Parquet)
│   ├── predictions/finetuning/<direction>/<zero_shot|finetuned|reference>/<target>/
│   ├── finetuning/<direction>/                # Pretrained checkpoint, learning curves (CSV) and metrics (JSON)
│   └── logs/                                  # stdout of every run
├── metrics/                       # Notebooks that aggregate the results into the paper tables
└── figures/                       # Figures produced from the notebooks
    └── <site>/shap/               # SHAP summary plots (one per target)
```

`<site>` is `alava` or `cartagena`, `<target>` is `temperature` or `rel_humidity`, and `<direction>` is `alava_to_cartagena` or `cartagena_to_alava`.

## Data

The data were acquired by the monitoring and climate-control systems of [NUTRICONTROL](https://nutricontrol.com/) in both installations. The acquisition and cleaning procedure is described in Carlavilla-Lázaro et al. (2026), *Computers and Electronics in Agriculture*, 113345.

**Raw data are not included** in this repository. The preprocessing scripts expect them under `data/raw/`, but they are only needed to regenerate the processed files; all experiments run from `data/processed/`.

| File | Description |
|---|---|
| `Alava_processed.csv` | Álava dataset after quality control, gap interpolation and feature generation |
| `Cartagena_comp0_processed.csv` | Same for Cartagena compartment C0 (indoor data joined with the outdoor weather station) |
| `Alava_processed_aligned.csv`, `Cartagena_comp0_processed_aligned.csv` | Both datasets restricted to their 66,237 common timestamps, with every column suffixed by site; used in the fine-tuning experiments |

All files are comma-separated, indexed by the `dateTime` column (15-minute resolution). Gaps from discarded days (maintenance, sensor outages, outliers) are kept as missing timestamps and are not filled. Column names follow the pattern `<VARIABLE>_<site>`, where the site suffix is `_Alava`, `_Cartagena_comp0` (indoor) or `_Cartagena_meteo` (outdoor station). The main variables are:

| Column prefix | Variable | Unit |
|---|---|---|
| `DB_TEMP_INT` | Indoor dry-bulb temperature (target) | °C |
| `REL_HUM_INT` | Indoor relative humidity (target) | % |
| `DB_TEMP_EXT`, `REL_HUM_EXT` | Outdoor temperature and relative humidity | °C, % |
| `RADIATION_MEAN/MIN/MAX` | Solar radiation | W/m² |
| `WIND_VELOCITY_*`, `WIND_DIRECTION` | Wind speed and direction | m/s, ° |
| `PIPE_TEMP`, `FLUID_TEMP_*` | Heating pipe / fluid temperature (Álava) | °C |
| `WINDOW_*`, `SHADE_SCREEN` | Window and thermal-screen positions | % |
| `VENTIL_TEMP`, `ACT_FORCED_VENT_STAGE` | Ventilation setpoint and forced-ventilation stage | °C, – |
| `WB_TEMP`, `VPD`, `HUM`, `WATER_DEF` | Wet-bulb temperature, vapor pressure deficit, absolute humidity, water deficit (Cartagena) | °C, kPa, g/m³, g/m³ |
| `hour`, `dayofweek`, `month`, `month_sin`, `month_cos`, `is_day` | Calendar features; `is_day` is computed astronomically | – |
| `*_lag2`, `*_lag4` | Lagged copies of a variable (2 and 4 steps) | – |
| `*_INPUT_*` | Copy of a target variable used as model input | – |

The exact input variables of each model are listed in the `input_variables` field of its result JSON and in Appendix A of the paper.

## Methodology

This section summarizes how the experiments are set up in the code. The paper gives the full rationale.

### Forecasting task

Each model receives a window of past observations and predicts the next **24 steps (6 hours)** of the target variable at 15-minute resolution. The two targets, indoor temperature (`DB_TEMP_INT`) and indoor relative humidity (`REL_HUM_INT`), are evaluated separately. The prediction horizon is the same in every experiment, including fine-tuning.

### Input configurations

Two sets of input variables are used:

- **Configuration A (architecture comparison).** Each architecture uses the site-specific variables that improved its accuracy, including indoor, outdoor and actuator-related signals such as pipe temperature. The selection differs by model family (see the table below and Appendix A of the paper).
- **Configuration B (cross-greenhouse transfer).** Only variables available at both sites are used, so that one model can run on either greenhouse: indoor temperature and relative humidity, outdoor temperature and relative humidity, mean solar radiation, and the calendar features `month_sin`, `month_cos` and `is_day`. These experiments read the `*_aligned.csv` files.

The input variables of every run are also stored in the `input_variables` field of its result JSON.

### Data split and scaling

Every dataset is split in the same way:

1. **Test set.** The last 30% of the records, in chronological order. No model sees it during training, so the test period lies strictly after the training period.
2. **Training and validation.** The first 70% is divided into alternating blocks of whole days: a block of training days followed by a block of validation days, repeated over the whole period. Block sizes are set in each script (`train_block_days` / `val_block_days`):

   | Models | Train / validation days |
   |---|---|
   | LSTM, GRU, CNN-LSTM, DLinear, NLinear | 3 / 2 (4 / 1 for the Álava LSTM and CNN-LSTM humidity models) |
   | PatchTST, iTransformer (including fine-tuning) | 14 / 7 |

   Input/output sequences are built inside each block, so no sequence spans two blocks.
3. **Scaling.** Min-max normalization is fitted on the training blocks only and then applied unchanged to the validation and test data.

### Architectures

| Family | Model | Input window | Inputs | Main hyperparameters |
|---|---|---|---|---|
| Recurrent | LSTM, GRU | 12 steps (3 h) | Multivariate with explicit lags (2 and 4 steps); separate model per target | 1 layer, 64 hidden units |
| Convolutional-recurrent | CNN-LSTM | 12 steps (3 h) | Same as LSTM/GRU | One 1D convolution (32 channels, kernel 3) + 1 LSTM layer, 64 hidden units |
| Linear | DLinear | 96 steps (24 h) | Univariate (target history only) | Moving-average kernel 49 |
| Linear | NLinear | 96 steps (24 h) | Univariate (target history only) | – |
| Transformer | PatchTST | 288 steps (3 days) | Multivariate, channel-independent; both targets in one model | Patch length 24, patch stride 12, `d_model` 64, 4 heads, 3 encoder layers, feed-forward 128, dropout 0.1 |
| Transformer | iTransformer | 96 steps (24 h) | Multivariate, attention across variables; both targets in one model | `d_model` 128, 8 heads, 3 encoder layers, feed-forward 128, dropout 0.1, instance normalization disabled |

The iTransformer hyperparameters were tuned on Álava and reused unchanged for Cartagena and for all fine-tuning experiments.

### Training

All models share the same training procedure:

- AdamW optimizer with the MSE loss and batch size 128.
- Initial learning rate 1e-4 (1e-3 for DLinear and NLinear), halved by `ReduceLROnPlateau` after 4 epochs without improvement in the validation loss (minimum 1e-6).
- Up to 100 epochs, with early stopping after 10 epochs without improvement in the validation loss. In the architecture-comparison scripts, the model is evaluated with the weights of the last epoch it trained. The transfer scripts instead restore the weights of the best validation epoch.
- Training sequences are generated with stride 1 (overlapping windows) and shuffled.

### Evaluation

- **Test predictions.** The trained model runs over the test set with consecutive, non-overlapping forecast windows (step = 24), so every test timestamp is predicted once. The predictions are converted back to physical units and saved as Parquet files.
- **Metrics.** R², MSE, RMSE, MAE and MAPE, computed over the full 24-step horizon. The paper does not report R² for relative humidity in Cartagena, because that variable stays close to saturation for long periods, which makes R² hard to interpret.
- **Repetitions.** Each configuration is run several times without a fixed seed, so each run starts from a different random initialization, and results are reported as mean ± SD: 5 runs for every architecture, and 3 runs for the transfer comparison (see `REPEATS` in `00_run_scripts.py`).

### Cross-greenhouse transfer

The transfer experiments use iTransformer with Configuration B, in both directions (Álava → Cartagena C0 and Cartagena C0 → Álava). The target site's last 30% is always the fixed test set and is never used for training or fine-tuning.

1. **Pretraining.** The model is trained from scratch on the source site's first 70%, following the procedure above.
2. **Experiment A: convergence curve** (`01_…` / `03_…`). The pretrained model is fine-tuned on all the target site's training data for 150 epochs without early stopping. Training and validation losses are logged at every epoch and test metrics every 5 epochs. These curves show how many epochs fine-tuning needs to converge.
3. **Experiment B: learning curve** (`01_…` / `03_…`). Fine-tuning is repeated from the same pretrained checkpoint with an increasing number of target training days (steps of 7 days, 3 seeds each), with early stopping. This shows how test error changes with the amount of target data.
4. **Experiment C: comparison** (`02_…` / `04_…`). Three conditions are evaluated on the same target test set:
   - **Direct transfer (zero-shot).** The pretrained model is applied without any adaptation.
   - **Fine-tuned.** The pretrained model is fine-tuned on all of the target site's training data (learning rate 1e-4, up to 100 epochs, early stopping with patience 10). The weights from the best validation epoch are kept.
   - **Target-only (reference).** A model with the same inputs is trained from scratch on the target site only, as the baseline for a dedicated model.

### Feature importance

The feature-importance analysis (`scripts/01_preprocessing/04_shap_feature_importance.py`) does not depend on the forecasting models. For each installation and target, a random-forest regressor is trained on the first 70% of the processed series (chronological split), and SHAP values are computed with `TreeExplainer` on a 5,000-sample subset of the remaining 30%. The `*_INPUT_*` columns, which are exact copies of the targets, and the lagged features are excluded. With `REMOVE_PROXY = True` (the default), the climate actuators that react to the indoor climate are also dropped: the window and shade-screen positions in Álava, and the window positions in Cartagena C0. One SHAP summary (beeswarm) plot per target is saved under `figures/<site>/shap/`.

Its main outcome was to identify window and thermal-screen positions as proxies of indoor temperature; see Sections 3.4 and 4.1 of the paper.

## Reproducing the results

### 1. Environment

The results were produced with Python 3.13 and PyTorch 2.14.0 (CUDA 13.0) on Windows 11 with an NVIDIA GeForce RTX 5070 Laptop GPU.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# PyTorch with CUDA 13.0 (use the CPU build from https://pytorch.org/get-started/locally/ if no GPU is available)
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

All scripts use relative paths and must be run from the repository root.

### 2. Preprocessing (optional)

Only needed if you have the raw data in `data/raw/`; the processed files are already included.

```bash
python scripts/01_preprocessing/01_preprocess_alava.py
python scripts/01_preprocessing/02_preprocess_cartagena_comp0.py
python scripts/01_preprocessing/03_align_alava_cartagena.py
```

The SHAP feature-importance analysis runs from the processed files, so it does not need the raw data:

```bash
python scripts/01_preprocessing/04_shap_feature_importance.py
```

### 3. Experiments

Run every experiment with the configured number of repetitions:

```bash
python 00_run_scripts.py
```

The runner calls the interpreter at `.venv/Scripts/python.exe` (Windows layout); edit `PYTHON_PATH` in `00_run_scripts.py` if your environment is elsewhere. Repetitions per architecture are set in `REPEATS`, and the stdout of each run is written to `results/logs/`.

A single experiment can also be run on its own, e.g.:

```bash
python scripts/02_experiments/alava/10_itransformer_alava.py
```

Each run appends an entry to the corresponding JSON in `results/experiments/` (configuration, input variables, architecture, training history and test metrics) and saves its test predictions as a Parquet file in `results/predictions/`. PatchTST and iTransformer predict both targets from a single model, so one run writes both the `temperature` and the `rel_humidity` results. The linear models (DLinear, NLinear) also train both targets in one script, as separate univariate models.

The fine-tuning scripts are organized in pairs for each direction:

| Script | Content |
|---|---|
| `01_…_alava_cartagena_finetuning_curves.py` / `03_…_cartagena_alava_…` | Pretraining on the source site, convergence curve of fine-tuning (epochs) and learning curve as a function of the number of target days. Output: `results/finetuning/<direction>/csv/` and the pretrained checkpoint |
| `02_…_alava_cartagena_direct_comparison.py` / `04_…_cartagena_alava_…` | Direct transfer vs. fine-tuning vs. target-only model. Output: `results/finetuning/<direction>/<target>/*.json` and Parquet predictions |

### 4. Tables and figures

The notebooks in `metrics/` read the JSON, CSV and Parquet results and compute the tables reported in the paper (architecture comparison per site and target, and fine-tuning experiments per direction). They must be run from the `metrics/` directory. The SHAP summary plots in `figures/<site>/shap/` are produced by the preprocessing script `04_shap_feature_importance.py` (see step 2), not by the notebooks.

## Acknowledgements

This work was developed within the SERGIoT project (CPP2023-010458), funded by MICIU/AEI/10.13039/501100011033 and FEDER (EU), and by the Universitat Politècnica de València through the Predoctoral Research Program (Subprogram 1, Grant PAID-01-24). Data were provided by NUTRICONTROL.

## Contact

Jaime Carlavilla-Lázaro — jcarlaz@etsinf.upv.es
Department of Computer Engineering (DISCA), Universitat Politècnica de València, Spain.
