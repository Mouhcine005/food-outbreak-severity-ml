# 🦠 Food Outbreak Severity Prediction

Predicting the severity level of foodborne disease outbreaks using Machine Learning,
based on CDC surveillance data (1998–2015).

## Problem Statement
Given epidemiological data about a food outbreak (pathogen, food type, location,
season), predict whether the outbreak severity is **Low**, **Moderate**, or **High**.

## Dataset
- Source: CDC Foodborne Disease Outbreak Surveillance System
- Period: 1998–2015
- Size: ~19,000 outbreak events

## Project Structure
food-outbreak-severity-ml/
│
├── data/
│   ├── raw/               ← original CSV goes here, never modified
│   ├── processed/         ← cleaned/engineered data saved here
│   └── external/          ← any extra datasets you add later
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_interpretability.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocess.py      ← cleaning functions
│   ├── features/
│   │   └── build_features.py  ← feature engineering functions
│   ├── models/
│   │   └── train.py           ← training & evaluation functions
│   └── visualization/
│       └── plots.py           ← reusable plot functions
│
├── reports/
│   └── figures/           ← saved plots for the report
│
├── models/                ← saved trained models (.pkl / .joblib)
│
├── requirements.txt
├── README.md
└── .gitignore

## How to Run
```bash
git clone ...
pip install -r requirements.txt
jupyter notebook
```

## Author
Mouhcine005