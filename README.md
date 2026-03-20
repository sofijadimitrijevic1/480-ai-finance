# Financial Fraud Detection 

This is a project for **CSC 480 Artificial Intelligence** at Cal Poly SLO. 

**Instructor:** Rodrigo Canaan
**Team members:** Sofija Dimitrijevic, Karen Severson, Sabrina Huang, Glenn Carvalho, Anissa Soungpanya

## External resources and credits

This project uses or is based on the following external resources:

- Kaggle dataset: `computingvictor/transactions-fraud-datasets`
- Python libraries: PyTorch, scikit-learn, pandas, numpy, matplotlib, imbalanced-learn, kagglehub

## Repository contents

- `model_comparison.ipynb` – main notebook for loading data, feature engineering, training models, and comparing results
- `preprocessing.py` – preprocessing pipeline for cleaning, feature engineering, scaling, and optional SMOTE
- `requirements.txt` – Python dependencies
- `download_data.sh` – helper script to download the dataset

## Installation

1. Create and activate a virtual environment.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

## Dataset setup

Either let the notebook download the dataset automatically with `kagglehub`, or set a local dataset path manually.

### Option A: automatic download
Run the notebook or script directly. If the dataset is not found locally, it will be downloaded automatically.

### Option B: local dataset directory
Place the dataset inside:

```text
data/transactions-fraud-datasets/
```

or set:

```bash
export FRAUD_DATA_DIR=/path/to/transactions-fraud-datasets
```

## Main use cases

### 1. Run the preprocessing pipeline
```bash
python preprocessing.py
```

This generates processed train/test arrays in:

```text
processed_data/
```

### 2. Run the main notebook
Open the notebook and run all cells:

```bash
jupyter notebook model_comparison.ipynb
```

The notebook performs:
- transaction loading and cleaning,
- temporal client-based splitting,
- tabular encoding and scaling,
- sequence window construction,
- baseline model training,
- CNN, CNN-RNN, and LSTM training,
- validation threshold selection,
- metric comparison on validation and test sets,
- diagnostic plots for probability distributions and threshold sensitivity.

## Important experiment parameters

These can be changed in the configuration cell near the top of the notebook:

- `N_CLIENTS` – number of sampled clients
- `SEQ_LEN` – sequence window length
- `MAX_WINDOWS_TRAIN`, `MAX_WINDOWS_VAL`, `MAX_WINDOWS_TEST` – max sampled sequence windows
- `BATCH_SIZE`
- `EPOCHS`
- `LEARNING_RATE`
- `POS_WEIGHT_CAP`

## Reproducing the main reported results

To reproduce the model comparison reported in the project:

1. Open `model_comparison.ipynb`
2. Run all cells from top to bottom
3. Use the final `df_compare` table as the main summary table
4. Use the grid search table (`df_grid`) if you want to reproduce hyperparameter search results
5. Use the diagnostic plots section for threshold sensitivity and probability clustering analysis