import os
import sys
import warnings
import joblib
import optuna
import numpy as np
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

NUMERIC_FEATURES = [
    "Year", "Month_num", "food_known",
    "location_risk", "is_peak_month", "state_outbreak_rate"
]
CATEGORICAL_LOW = [
    "State", "Location", "Status",
    "Species_Risk", "Food_Category", "Season"
]
CATEGORICAL_HIGH = ["species_x_location"]


def build_preprocessor():
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


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight":      trial.suggest_categorical(
                                "class_weight", ["balanced", "1-1-2", "1-1-3", "1-15-2", "1-12-25"]
                             ),
    }

    # Convert class_weight string to dict
    cw_map = {
        "balanced":  "balanced",
        "1-1-2":     {0:1, 1:1,   2:2},
        "1-1-3":     {0:1, 1:1,   2:3},
        "1-15-2":    {0:1, 1:1.5, 2:2},
        "1-12-25":   {0:1, 1:1.2, 2:2.5},
    }
    cw = cw_map[params.pop("class_weight")]

    clf = LGBMClassifier(
        **params,
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    pipeline = ImbPipeline([
        ("pre", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    return f1_score(y_val, y_pred, average="macro")


def run_tuning(X_train, y_train, X_val, y_val, n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest Macro F1: {study.best_value:.4f}")
    print(f"Best params:  {study.best_params}")

    # Retrain best model on train+val combined
    best_params = study.best_params.copy()
    cw_map = {
        "balanced":  "balanced",
        "1-1-2":     {0:1, 1:1,   2:2},
        "1-1-3":     {0:1, 1:1,   2:3},
        "1-15-2":    {0:1, 1:1.5, 2:2},
        "1-12-25":   {0:1, 1:1.2, 2:2.5},
    }
    cw = cw_map[best_params.pop("class_weight")]

    best_clf = LGBMClassifier(
        **best_params,
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    X_trainval = np.vstack([X_train, X_val]) if hasattr(X_train, 'values') == False else \
                 __import__('pandas').concat([X_train, X_val])
    y_trainval = __import__('numpy').concatenate([y_train, y_val])

    best_pipeline = ImbPipeline([
        ("pre", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", best_clf),
    ])
    best_pipeline.fit(X_trainval, y_trainval)

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "LightGBM_tuned.joblib")
    joblib.dump(best_pipeline, save_path)
    print(f"Saved tuned model → {save_path}")

    return best_pipeline, study