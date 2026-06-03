import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_feature_names(pipeline):
    pre = pipeline.named_steps["pre"]
    num_features = list(pre.transformers_[0][2])
    ohe = pre.transformers_[1][1]
    cat_low_features = list(ohe.get_feature_names_out(pre.transformers_[1][2]))
    cat_high_features = list(pre.transformers_[2][2])
    return num_features + cat_low_features + cat_high_features


def get_transformed_X(pipeline, X):
    pre = pipeline.named_steps["pre"]
    X_transformed = pre.transform(X)
    feature_names = get_feature_names(pipeline)
    return pd.DataFrame(X_transformed, columns=feature_names)


def compute_shap_values(pipeline, X, max_samples=500):
    clf = pipeline.named_steps["clf"]
    X_transformed = get_transformed_X(pipeline, X)
    if len(X_transformed) > max_samples:
        X_transformed = X_transformed.sample(max_samples, random_state=42)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)
    # For binary LightGBM shap_values is list [class0, class1] - take class1 (High)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    return sv, X_transformed


def plot_shap_summary(pipeline, X, max_samples=500):
    sv, X_transformed = compute_shap_values(pipeline, X, max_samples)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_transformed, plot_type="bar", max_display=15, show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "reports", "figures", "shap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved ->", save_path)
    return sv, X_transformed


def plot_shap_beeswarm(pipeline, X, max_samples=500):
    sv, X_transformed = compute_shap_values(pipeline, X, max_samples)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_transformed, plot_type="dot", max_display=15, show=False)
    plt.title("SHAP Beeswarm - Feature Impact Direction", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "reports", "figures", "shap_beeswarm.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved ->", save_path)


def plot_waterfall_examples(pipeline, X, y_true, y_proba, threshold=0.35, n=3):
    clf = pipeline.named_steps["clf"]
    X_transformed = get_transformed_X(pipeline, X)
    explainer = shap.TreeExplainer(clf)

    y_pred = (y_proba >= threshold).astype(int)
    correct_high_idx = np.where((y_true.values == 1) & (y_pred == 1))[0][:n]

    for i, idx in enumerate(correct_high_idx):
        row = X_transformed.iloc[[idx]]
        sv_raw = explainer.shap_values(row)
        sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
        base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(shap.Explanation(
            values=sv[0],
            base_values=base,
            data=row.iloc[0].values,
            feature_names=list(row.columns)
        ), max_display=12, show=False)
        plt.title(f"Why predicted HIGH - example {i+1}", fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, "reports", "figures", f"shap_waterfall_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved ->", save_path)


def error_analysis(X, y_true, y_proba, threshold=0.35):
    y_pred = (y_proba >= threshold).astype(int)

    df = X.copy()
    df["y_true"] = y_true.values
    df["y_pred"] = y_pred
    df["y_proba"] = y_proba

    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]
    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]

    print(f"False Negatives (missed High): {len(fn)}")
    print(f"False Positives (false alarms): {len(fp)}")

    print("\n--- Missed High outbreaks (FN) - Location breakdown ---")
    print(fn["Location"].value_counts().head(8))

    print("\n--- Missed High outbreaks (FN) - Species_Risk breakdown ---")
    print(fn["Species_Risk"].value_counts().head(8))

    print("\n--- False alarms (FP) - Location breakdown ---")
    print(fp["Location"].value_counts().head(8))

    print("\n--- Avg proba for missed High cases ---")
    print(f"  Mean: {fn['y_proba'].mean():.3f}  Median: {fn['y_proba'].median():.3f}")

    return fn, fp