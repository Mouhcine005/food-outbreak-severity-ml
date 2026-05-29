import pandas as pd
import numpy as np


# ── 1. COLUMN DROPPING ──────────────────────────────────────────────────────

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with too much missing data or that cause leakage."""
    cols_to_drop = ["Ingredient", "Serotype/Genotype", "Hospitalizations", "Fatalities"]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")
    return df


# ── 2. DUPLICATES ───────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows ({before} → {after})")
    return df


# ── 3. LOCATION CLEANING ────────────────────────────────────────────────────

def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplify Location column:
    - Take only the first location from compound values (e.g. 'Restaurant; Home' → 'Restaurant')
    - Group rare locations into 'Other'
    - Fill missing with 'Unknown'
    """
    df = df.copy()

    # Take first location from compound values
    df["Location"] = df["Location"].str.split(";").str[0].str.strip()

    # Define top locations to keep (based on EDA plots)
    top_locations = [
        "Restaurant",
        "Private Home/Residence",
        "Catering Service",
        "Banquet Facility",
        "Fast Food Restaurant",
        "School/College/University",
        "Prison/Jail",
        "Nursing Home/Assisted Living Facility",
        "Grocery Store",
        "Camp",
        "Religious Facility",
        "Office/Indoor Worker"
    ]

    df["Location"] = df["Location"].apply(
        lambda x: x if x in top_locations else ("Unknown" if pd.isna(x) else "Other")
    )

    print(f"Location unique values after cleaning: {df['Location'].nunique()}")
    print(df["Location"].value_counts())
    return df


# ── 4. SPECIES RISK TIERS ───────────────────────────────────────────────────

def clean_species(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group Species into risk tiers based on EDA:
    - Norovirus: dominant in dataset, generally lower severity
    - Salmonella: strong second, moderate-high severity
    - High Risk: pathogens known for serious outcomes
    - Medium Risk: moderate pathogens
    - Unknown: missing
    - Other: everything else
    """
    df = df.copy()

    def assign_risk_tier(species):
        if pd.isna(species):
            return "Unknown"

        s = str(species).lower()

        if "norovirus" in s:
            return "Norovirus"
        elif "salmonella" in s:
            return "Salmonella"
        elif any(x in s for x in [
            "escherichia coli", "e. coli", "listeria", "shigella",
            "hepatitis", "vibrio", "clostridium botulinum"
        ]):
            return "High_Risk"
        elif any(x in s for x in [
            "clostridium perfringens", "staphylococcus", "bacillus",
            "campylobacter", "yersinia"
        ]):
            return "Medium_Risk"
        elif any(x in s for x in [
            "scombroid", "ciguatoxin", "chemical", "toxin", "mushroom",
            "histamine"
        ]):
            return "Toxin"
        else:
            return "Other"

    df["Species_Risk"] = df["Species"].apply(assign_risk_tier)
    df = df.drop(columns=["Species"])

    print(f"Species_Risk distribution:")
    print(df["Species_Risk"].value_counts())
    return df


# ── 5. FOOD CLEANING ────────────────────────────────────────────────────────

def clean_food(df: pd.DataFrame) -> pd.DataFrame:
    """
    Food has 47% missing and 3127 unique values — too many to use raw.
    Strategy:
    - Create binary flag: food_known (1 if food was identified, 0 if not)
    - Group into broad food categories
    """
    df = df.copy()

    # Binary flag
    df["food_known"] = df["Food"].notna().astype(int)

    def categorize_food(food):
        if pd.isna(food):
            return "Unknown"

        f = str(food).lower()

        if any(x in f for x in ["chicken", "turkey", "beef", "pork", "meat",
                                  "ground beef", "hamburger", "steak", "lamb"]):
            return "Meat_Poultry"
        elif any(x in f for x in ["fish", "salmon", "tuna", "shrimp", "oyster",
                                    "seafood", "crab", "lobster", "scombroid"]):
            return "Seafood"
        elif any(x in f for x in ["egg", "custard", "mayonnaise", "mayo"]):
            return "Eggs_Dairy"
        elif any(x in f for x in ["salad", "lettuce", "vegetable", "fruit",
                                    "tomato", "spinach", "sprout"]):
            return "Produce"
        elif any(x in f for x in ["rice", "pasta", "bread", "sandwich",
                                    "pizza", "noodle", "grain", "stuffing"]):
            return "Grains_Starch"
        elif any(x in f for x in ["multiple", "various"]):
            return "Multiple_Foods"
        else:
            return "Other"

    df["Food_Category"] = df["Food"].apply(categorize_food)
    df = df.drop(columns=["Food"])

    print(f"Food_Category distribution:")
    print(df["Food_Category"].value_counts())
    print(f"food_known: {df['food_known'].sum()} known, {(df['food_known']==0).sum()} unknown")
    return df


# ── 6. MONTH ENCODING ───────────────────────────────────────────────────────

def encode_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Month as:
    - Month_num: 1-12 (numerical order)
    - Season: based on EDA pattern (Summer peak, December spike)
    """
    df = df.copy()

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["Month_num"] = df["Month"].map(month_map)

    def get_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["Season"] = df["Month_num"].apply(get_season)
    df = df.drop(columns=["Month"])

    print(f"Season distribution:")
    print(df["Season"].value_counts())
    return df


# ── 7. STATUS CLEANING ──────────────────────────────────────────────────────

def clean_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplify compound Status values:
    'Confirmed; Confirmed' → 'Confirmed'
    'Suspected; Confirmed' → 'Mixed'
    Missing → 'Unknown'
    """
    df = df.copy()

    def simplify_status(s):
        if pd.isna(s):
            return "Unknown"
        parts = set([x.strip() for x in s.split(";")])
        if len(parts) == 1:
            return parts.pop()
        else:
            return "Mixed"

    df["Status"] = df["Status"].apply(simplify_status)
    print(f"Status distribution:")
    print(df["Status"].value_counts())
    return df


# ── 8. MASTER PIPELINE ──────────────────────────────────────────────────────

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all preprocessing steps in order."""
    print("=" * 50)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 50)

    df = drop_useless_columns(df)
    print()
    df = remove_duplicates(df)
    print()
    df = clean_location(df)
    print()
    df = clean_species(df)
    print()
    df = clean_food(df)
    print()
    df = encode_month(df)
    print()
    df = clean_status(df)

    print()
    print("=" * 50)
    print(f"PIPELINE COMPLETE — Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("=" * 50)

    return df