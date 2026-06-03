import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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


def build_stacking_pipeline():
    estimators = [
        ("lgbm", LGBMClassifier(
            n_estimators=466,
            learning_rate=0.0192,
            num_leaves=33,
            max_depth=12,
            min_child_samples=65,
            subsample=0.56,
            colsample_bytree=0.54,
            class_weight={0:1, 1:1.5, 2:2},
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
        ("xgb", XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )),
        ("rf", RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
    ]

    meta_clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    pipeline = ImbPipeline([
        ("pre", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", stacking),
    ])

    return pipeline


def run_stacking(X_train, y_train, X_val, y_val, X_test, y_test):
    from src.models.evaluate import evaluate_model, plot_confusion_matrix

    print("Training stacking ensemble (this takes ~5-10 min)...")
    pipeline = build_stacking_pipeline()
    pipeline.fit(X_train, y_train)

    print("\n--- Validation ---")
    f1_val, y_pred_val = evaluate_model(pipeline, X_val, y_val, model_name="Stacking")
    plot_confusion_matrix(y_val, y_pred_val, model_name="Stacking — Val")

    print("\n--- Test ---")
    f1_test, y_pred_test = evaluate_model(pipeline, X_test, y_test,
                                           model_name="Stacking", dataset_label="TEST SET")
    plot_confusion_matrix(y_test, y_pred_test, model_name="Stacking — Test")

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "stacking.joblib")
    joblib.dump(pipeline, save_path)
    print(f"\nSaved → {save_path}")

    return pipeline, f1_val, f1_test