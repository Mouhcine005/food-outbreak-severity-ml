import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)


CLASS_NAMES = ["Low", "Moderate", "High"]


def evaluate_model(model, X, y, model_name="Model", dataset_label="Validation"):
    """
    Print classification report and return macro F1 + predictions.
    """
    y_pred = model.predict(X)
    macro_f1 = f1_score(y, y_pred, average="macro")

    print(f"\n{'='*55}")
    print(f"  {model_name}  [{dataset_label}]")
    print(f"{'='*55}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print()
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))

    return macro_f1, y_pred


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Plot a labelled confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def compare_models(scores: dict):
    """
    Print a ranked summary table of validation macro F1 scores.
    scores: {model_name: macro_f1}
    """
    print(f"\n{'='*40}")
    print("  MODEL COMPARISON (Val Macro F1)")
    print(f"{'='*40}")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(ranked, 1):
        marker = "  ◀ BEST" if rank == 1 else ""
        print(f"  {rank}. {name:<22} {score:.4f}{marker}")
    print(f"{'='*40}")
    return ranked[0][0]  # returns best model name