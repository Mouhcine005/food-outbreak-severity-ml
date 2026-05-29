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

# ── FEATURE ENGINEERING FUNCTIONS ───────────────────────────────────────────

def add_log_illnesses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log transform of Illnesses.
    Compresses the heavy right skew we saw in EDA.
    Using log1p (log(x+1)) to safely handle any zero values.
    """
    df = df.copy()
    df["illnesses_log"] = np.log1p(df["Illnesses"])
    print(f"illnesses_log — min: {df['illnesses_log'].min():.2f}, max: {df['illnesses_log'].max():.2f}")
    return df


def add_location_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a numerical risk score to each location.
    Based on EDA findings — locations with vulnerable populations
    or mass gathering potential score higher.
    Scale: 1 (lowest risk) to 5 (highest risk)
    """
    df = df.copy()

    location_risk_map = {
        "Nursing Home/Assisted Living Facility": 5,
        "Prison/Jail":                           5,
        "School/College/University":             4,
        "Catering Service":                      4,
        "Banquet Facility":                      4,
        "Camp":                                  4,
        "Religious Facility":                    3,
        "Restaurant":                            3,
        "Fast Food Restaurant":                  3,
        "Grocery Store":                         2,
        "Office/Indoor Workplace":               2,
        "Private Home/Residence":                2,
        "Other":                                 2,
        "Unknown":                               1,
    }

    df["location_risk"] = df["Location"].map(location_risk_map).fillna(2)
    print(f"location_risk distribution:")
    print(df["location_risk"].value_counts().sort_index())
    return df


def add_peak_month_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag for peak outbreak months.
    From EDA: May(5), June(6) are summer peaks.
    December(12) is the holiday spike.
    March(3), April(4) are spring rise.
    """
    df = df.copy()

    peak_months = [3, 4, 5, 6, 12]
    df["is_peak_month"] = df["Month_num"].isin(peak_months).astype(int)

    print(f"is_peak_month — peak: {df['is_peak_month'].sum()}, non-peak: {(df['is_peak_month']==0).sum()}")
    return df


def add_species_location_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction feature between Species_Risk and Location.
    A High_Risk pathogen in a Nursing Home is far more dangerous
    than the same pathogen in a private home.
    Creates a combined string category.
    """
    df = df.copy()

    df["species_x_location"] = df["Species_Risk"] + "_" + df["Location"]
    print(f"species_x_location — unique combinations: {df['species_x_location'].nunique()}")
    return df


def add_state_outbreak_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    How many total outbreaks each state has in the dataset.
    States with more outbreaks (Florida, California) may have
    systemic food safety differences worth encoding.
    Normalized to 0-1 scale.
    """
    df = df.copy()

    state_counts = df["State"].value_counts()
    state_rate = (state_counts - state_counts.min()) / (state_counts.max() - state_counts.min())
    df["state_outbreak_rate"] = df["State"].map(state_rate)

    print(f"state_outbreak_rate — min: {df['state_outbreak_rate'].min():.3f}, max: {df['state_outbreak_rate'].max():.3f}")
    return df


# ── MASTER PIPELINE ──────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in order."""
    print("=" * 50)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 50)

    df = add_log_illnesses(df)
    print()
    df = add_location_risk(df)
    print()
    df = add_peak_month_flag(df)
    print()
    df = add_species_location_interaction(df)
    print()
    df = add_state_outbreak_rate(df)

    print()
    print("=" * 50)
    print(f"FEATURE ENGINEERING COMPLETE — Final shape: {df.shape}")
    print(f"New columns added: illnesses_log, location_risk, is_peak_month, species_x_location, state_outbreak_rate")
    print("=" * 50)

    return df