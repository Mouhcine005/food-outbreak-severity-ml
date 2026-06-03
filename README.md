# 🦠 Food Outbreak Severity Prediction

> **Machine learning pipeline to predict U.S. foodborne outbreak severity using CDC surveillance data.**  
> Binary LightGBM + Optuna + SMOTE + SHAP | **ROC-AUC: 82.7% | Macro F1: 69.6% | Accuracy: 82.3% | High Recall: 50.1%**

---

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)

---

## Overview

This project builds a competition-level ML system to classify U.S. foodborne outbreaks as **High severity** or **Not High** using only features available at or before detection — no outcome leakage.

**The core challenge:** The CDC dataset is heavily imbalanced (only ~15% High-severity events), has significant missing data in key columns (Food: 47%, Species: 35%), and requires careful feature engineering to extract signal without leaking the target variable.

**The solution:** A binary LightGBM classifier with SMOTE oversampling, 100-trial Optuna hyperparameter search, custom decision threshold tuning (t=0.35), and SHAP interpretability analysis.

---

## Results

### Model Comparison (Validation Set — Macro F1)

| Model | Val Macro F1 |
|---|---|
| **LightGBM** | **57.3%** 🥇 |
| XGBoost | 56.6% |
| Logistic Regression | 54.9% |
| Random Forest | 53.6% |

### Improvement Journey

| Stage | Macro F1 |
|---|---|
| Multiclass LightGBM baseline | 57.3% |
| Multiclass + class weight tuning | ~57–58% |
| Multiclass + Optuna 100 trials | ~59–60% |
| Stacking Ensemble (LGBM + XGB + RF → LR) | ~57–58% |
| **→ Pivot: Binary LightGBM (High vs Not High)** | **~63%** |
| + Threshold tuning (t=0.25) | improved High recall |
| + Optuna binary tuning (100 trials) | ~68–69% |
| **+ Final threshold optimization (t=0.35)** | **69.6% ✓** |

### Final Test Set Metrics

| Metric | Score |
|---|---|
| **ROC-AUC** | **82.7%** |
| **Macro F1** | **69.6%** |
| **Accuracy** | **82.3%** |
| **High Severity Recall (F1)** | **50.1%** |
| Decision Threshold | 0.35 |

---

## Dataset

- **Source:** CDC Foodborne Outbreak Surveillance Database
- **Time range:** 1998–2015 (18 years)
- **Size:** 19,119 outbreaks, 12 columns
- **Geography:** 55 U.S. states and territories

### Severity Labels (derived from Illnesses)

| Class | Rule | Threshold | Count | Share |
|---|---|---|---|---|
| Low | ≤ 50th percentile | ≤ 8 illnesses | 10,163 | 53.2% |
| Moderate | 50th–85th percentile | 9–30 illnesses | 6,152 | 32.2% |
| High | > 85th percentile | > 30 illnesses | 2,804 | 14.7% |

> ⚠️ **Leakage guard:** `Illnesses`, `illnesses_log`, `Hospitalizations`, and `Fatalities` are **never** used as model features — they are outcome variables known only after an outbreak concludes.

---

## Pipeline

```
Raw CDC Data (19,119 rows)
       │
       ▼
[1-EDA.ipynb]
   Distributions, temporal trends, top pathogens/locations, seasonality
       │
       ▼
[2-target_engineering.ipynb]
   Severity labels from Illnesses percentiles (50th / 85th)
       │
       ▼
[3-preprocessing.ipynb]  →  src/data/preprocess.py
   • Drop leaky/useless columns (Ingredient, Serotype, Hosp, Fatal)
   • Location: compound split + group to 14 categories
   • Species: 3,500+ values → 6 risk tiers (Norovirus, Salmonella, High_Risk, Medium_Risk, Toxin, Other)
   • Food: binary flag (food_known) + 7 Food_Category groups
   • Month: → Month_num (1-12) + Season (Winter/Spring/Summer/Fall)
   • Status: compound parsing → Confirmed / Suspected / Mixed / Unknown
       │
       ▼
[4-feature_engineering.ipynb]  →  src/features/build_features.py
   • illnesses_log      — log1p(Illnesses), excluded from X (leakage reference only)
   • location_risk      — ordinal risk score 1–5 per location type
   • is_peak_month      — binary flag for Mar/Apr/May/Jun/Dec
   • species_x_location — interaction of Species_Risk × Location (90 combos)
   • state_outbreak_rate — normalized state-level outbreak frequency
       │
       ▼
[5-modeling.ipynb]  →  src/models/
   Phase 1: 4-model multiclass baseline (LGBM wins at 57.3% Macro F1)
   Phase 2: Class weight tuning, Optuna multiclass, stacking (ceiling ~59-60%)
   Phase 3: PIVOT to binary → threshold sweep → Optuna binary → BEST: 69.6% F1
       │
       ▼
[6-interpretability.ipynb]  →  src/models/explain.py
   SHAP summary, beeswarm, waterfall plots, error analysis
```

### Feature Set

**Numeric** (StandardScaler): `Year`, `Month_num`, `food_known`, `location_risk`, `is_peak_month`, `state_outbreak_rate`

**Categorical — low cardinality** (OneHotEncoder): `State`, `Location`, `Status`, `Species_Risk`, `Food_Category`, `Season`

**Categorical — high cardinality** (OrdinalEncoder): `species_x_location` (90 unique combinations)

---

## Project Structure

```
food-outbreak-severity-ml/
├── data/
│   ├── raw/
│   │   └── outbreaks.csv                  # Original CDC data
│   └── processed/
│       ├── outbreaks_with_target.csv      # After step 2
│       ├── outbreaks_cleaned.csv          # After step 3
│       └── outbreaks_featured.csv         # After step 4 (model input)
│
├── notebooks/
│   ├── 1-EDA.ipynb                        # Exploratory data analysis
│   ├── 2-target_engineering.ipynb         # Severity label creation
│   ├── 3-preprocessing.ipynb              # Data cleaning pipeline
│   ├── 4-feature_engineering.ipynb        # Feature engineering
│   ├── 5-modeling.ipynb                   # All modeling experiments
│   └── 6-interpretability.ipynb           # SHAP + error analysis
│
├── src/
│   ├── data/
│   │   └── preprocess.py                  # 8-step preprocessing pipeline
│   ├── features/
│   │   └── build_features.py              # Feature engineering + severity labeling
│   └── models/
│       ├── train.py                       # Model definitions, preprocessor, train_all()
│       ├── tune.py                        # Optuna hyperparameter tuning
│       ├── stack.py                       # Stacking ensemble (LGBM + XGB + RF → LR)
│       ├── evaluate.py                    # Metrics, confusion matrix, comparison
│       └── explain.py                     # SHAP plots and error analysis
│
├── reports/
│   └── figures/                           # All saved plots (PNG)
│       ├── 01_distributions.png
│       ├── 02_outbreaks_over_time.png
│       ├── 02_target_distribution.png
│       ├── 03_top_pathogens.png
│       ├── 04_feature_correlations.png
│       ├── 04_log_transform.png
│       ├── 04_top_locations.png
│       ├── 05_outbreaks_by_month.png
│       ├── 06_correlation.png
│       ├── shap_summary.png
│       ├── shap_beeswarm.png
│       └── shap_waterfall_[1-3].png
│
├── results/
│   └── model_comparison.csv               # Validation Macro F1 for all baselines
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/food-outbreak-severity-ml.git
cd food-outbreak-severity-ml

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Pipeline (in order)

```bash
jupyter notebook
```

Open and run notebooks 1 through 6 in sequence:

| Notebook | What it does | Output |
|---|---|---|
| `1-EDA.ipynb` | Exploratory analysis | `reports/figures/` |
| `2-target_engineering.ipynb` | Creates severity labels | `data/processed/outbreaks_with_target.csv` |
| `3-preprocessing.ipynb` | Cleans the data | `data/processed/outbreaks_cleaned.csv` |
| `4-feature_engineering.ipynb` | Engineers features | `data/processed/outbreaks_featured.csv` |
| `5-modeling.ipynb` | Trains and evaluates all models | `results/`, `models/` |
| `6-interpretability.ipynb` | SHAP analysis | `reports/figures/shap_*.png` |

### 3. Use the Final Model Directly

```python
import joblib
import pandas as pd

# Load the saved model + threshold
artifact = joblib.load("models/LightGBM_binary_final.joblib")
pipeline  = artifact["pipeline"]
threshold = artifact["threshold"]  # 0.35

# Predict on new data (must match feature schema)
probas = pipeline.predict_proba(X_new)[:, 1]
predictions = (probas >= threshold).astype(int)  # 1 = High, 0 = Not High
```