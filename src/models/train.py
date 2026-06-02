import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── CONSTANTS ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "outbreaks_featured.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

NUMERIC_FEATURES = [
    "Year", "Month_num", "food_known",
    "location_risk", "is_peak_month", "state_outbreak_rate"
]

# species_x_location is HIGH cardinality (90 unique) — handled separately
CATEGORICAL_LOW = [
    "State", "Location", "Status",
    "Species_Risk", "Food_Category", "Season"
]
CATEGORICAL_HIGH = ["species_x_location"]   # OrdinalEncoder to avoid 90-column OHE explosion

TARGET = "Severity_code"

# These must NEVER appear in X — they are leakage or the target itself
LEAKAGE_COLS = ["Illnesses", "illnesses_log", "Severity", "Severity_code"]


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH):
    """Load the featured CSV and return X, y."""
    df = pd.read_csv(path)
    X = df.drop(columns=LEAKAGE_COLS)
    y = df[TARGET]
    print(f"Loaded: {path}  →  X{X.shape}  y{y.shape}")
    print(f"Class distribution:\n{y.value_counts().sort_index().to_string()}\n")
    return X, y


# ── SPLIT ────────────────────────────────────────────────────────────────────

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Stratified 70 / 15 / 15 split.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First cut off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split remainder into train/val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    for name, subset in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = subset.value_counts(normalize=True).sort_index().mul(100).round(1)
        print(f"{name} ({len(subset)} rows) — Low:{dist[0]}%  Mod:{dist[1]}%  High:{dist[2]}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── PREPROCESSOR ─────────────────────────────────────────────────────────────

def build_preprocessor():
    """
    ColumnTransformer:
      - Numeric      → StandardScaler
      - Low-card cat → OneHotEncoder (handle_unknown='ignore')
      - High-card cat (species_x_location, 90 vals) → OrdinalEncoder via OHE with max_categories
    """
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat_low",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_LOW),
            ("cat_high",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_HIGH),
        ],
        remainder="drop"
    )


# ── MODEL DEFINITIONS ────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="saga",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }


# ── PIPELINE BUILDER ─────────────────────────────────────────────────────────

def build_pipeline(clf, use_smote: bool = True):
    """
    Wrap preprocessor + optional SMOTE + classifier into a single pipeline.
    Uses imblearn Pipeline when SMOTE is enabled so SMOTE only runs on
    training folds (never leaks into val/test).
    """
    preprocessor = build_preprocessor()

    if use_smote:
        return ImbPipeline([
            ("pre", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("clf", clf),
        ])
    else:
        return Pipeline([
            ("pre", preprocessor),
            ("clf", clf),
        ])


# ── TRAINING ─────────────────────────────────────────────────────────────────

def train_all(
    X_train, y_train,
    use_smote: bool = True,
    save_models: bool = True,
) -> dict:
    """
    Train all models. Returns dict of {name: fitted_pipeline}.
    Saves .joblib files to models/ directory.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    trained = {}

    for name, clf in get_models().items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(clf, use_smote=use_smote)
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

        if save_models:
            path = os.path.join(MODELS_DIR, f"{name}.joblib")
            joblib.dump(pipeline, path)
            print(f"  Saved → {path}")

    print(f"\nAll models trained. Files in '{MODELS_DIR}/'")
    return trained


# ── LOAD SAVED MODEL ─────────────────────────────────────────────────────────

def load_model(name: str):
    """Load a saved model by name (e.g. 'XGBoost')."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Run train_all() first.")
    return joblib.load(path)