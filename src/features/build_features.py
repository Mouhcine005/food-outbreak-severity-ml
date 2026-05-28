import pandas as pd
import numpy as np

def define_severity(df: pd.DataFrame, low_pct: float = 0.50, high_pct: float = 0.85) -> pd.DataFrame:
    """
    Assigns a severity label to each outbreak based on Illnesses percentiles.

    Labels:
        Low      → Illnesses <= 50th percentile
        Moderate → 50th < Illnesses <= 85th percentile
        High     → Illnesses > 85th percentile

    Args:
        df       : raw dataframe (must contain 'Illnesses' column)
        low_pct  : upper bound percentile for 'Low' class
        high_pct : lower bound percentile for 'High' class

    Returns:
        df with new columns: 'Severity', 'Severity_code'
    """
    df = df.copy()

    low_thresh  = df["Illnesses"].quantile(low_pct)
    high_thresh = df["Illnesses"].quantile(high_pct)

    print(f"Illnesses thresholds:")
    print(f"  Low  (≤ {low_pct*100:.0f}th pct) : <= {low_thresh}")
    print(f"  High (> {high_pct*100:.0f}th pct) : >  {high_thresh}")
    print(f"  Moderate: everything in between\n")

    def label(x):
        if x <= low_thresh:
            return "Low"
        elif x <= high_thresh:
            return "Moderate"
        else:
            return "High"

    df["Severity"] = df["Illnesses"].apply(label)

    # Ordered encoding for potential use later
    order_map = {"Low": 0, "Moderate": 1, "High": 2}
    df["Severity_code"] = df["Severity"].map(order_map)

    return df, low_thresh, high_thresh


def print_class_distribution(df: pd.DataFrame):
    counts = df["Severity"].value_counts()
    pcts   = df["Severity"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"Count": counts, "Percent": pcts.round(1)})
    print("Class Distribution:")
    print(summary)
    return summary