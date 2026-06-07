import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Outbreak Severity Predictor",
    page_icon="🦠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.main { background-color: #0f1117; }

.metric-card {
    background: #1a1d27;
    border: 1px solid #2d3147;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.result-high {
    background: linear-gradient(135deg, #3d1a1a, #5c1f1f);
    border: 2px solid #e53935;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-nothigh {
    background: linear-gradient(135deg, #1a3d1a, #1f5c1f);
    border: 2px solid #43a047;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: 2px;
}
.prob-bar-container {
    background: #1a1d27;
    border-radius: 8px;
    height: 12px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6c7293;
    margin-bottom: 1rem;
    border-bottom: 1px solid #2d3147;
    padding-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    artifact = joblib.load("models/LightGBM_binary_final.joblib")
    return artifact["pipeline"], artifact["threshold"]

try:
    pipeline, threshold = load_artifact()
except Exception as e:
    st.error(f"❌ Could not load model: `models/LightGBM_binary_final.joblib`\n\n`{e}`")
    st.stop()


# ── Load reference data for state_outbreak_rate ───────────────────────────────
@st.cache_data
def load_reference():
    df = pd.read_csv("data/processed/outbreaks_featured.csv")
    # Rebuild the state_outbreak_rate map from training data
    from src.data.preprocess import run_preprocessing_pipeline
    raw = pd.read_csv("data/raw/outbreaks.csv")
    state_counts = df["State"].value_counts()
    state_rate = (state_counts - state_counts.min()) / (state_counts.max() - state_counts.min())
    return state_rate.to_dict(), sorted(df["State"].unique().tolist())

state_rate_map, all_states = load_reference()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🦠 Food Outbreak Severity")
st.markdown("**Binary classifier** — predicts whether a U.S. foodborne outbreak is `HIGH` severity or `NOT HIGH`  \nModel: LightGBM + Optuna + SMOTE · ROC-AUC: 82.7% · Macro F1: 69.6%")
st.markdown("---")

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-header">Outbreak Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        year = st.slider("Year", min_value=1998, max_value=2015, value=2008)
        month = st.selectbox("Month", [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ], index=5)
        state = st.selectbox("State", all_states)
        location = st.selectbox("Location", [
            "Restaurant", "Private Home/Residence", "Catering Service",
            "Banquet Facility", "Fast Food Restaurant",
            "School/College/University", "Prison/Jail",
            "Nursing Home/Assisted Living Facility", "Grocery Store",
            "Camp", "Religious Facility", "Office/Indoor Workplace", "Other"
        ])

    with col2:
        species_risk = st.selectbox("Pathogen Risk Tier", [
            "Norovirus", "Salmonella", "High_Risk",
            "Medium_Risk", "Toxin", "Other", "Unknown"
        ])
        food_category = st.selectbox("Food Category", [
            "Meat_Poultry", "Seafood", "Eggs_Dairy",
            "Produce", "Grains_Starch", "Multiple_Foods", "Other", "Unknown"
        ])
        status = st.selectbox("Etiology Status", [
            "Confirmed", "Suspected", "Mixed", "Unknown"
        ])
        food_known = st.radio("Food Vehicle Identified?", ["Yes", "No"], horizontal=True)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Severity", use_container_width=True, type="primary")


# ── Feature construction ──────────────────────────────────────────────────────
month_map = {
    "January":1,"February":2,"March":3,"April":4,
    "May":5,"June":6,"July":7,"August":8,
    "September":9,"October":10,"November":11,"December":12
}
def get_season(m):
    if m in [12,1,2]: return "Winter"
    if m in [3,4,5]:  return "Spring"
    if m in [6,7,8]:  return "Summer"
    return "Fall"

location_risk_map = {
    "Nursing Home/Assisted Living Facility": 5,
    "Prison/Jail": 5,
    "School/College/University": 4,
    "Catering Service": 4,
    "Banquet Facility": 4,
    "Camp": 4,
    "Religious Facility": 3,
    "Restaurant": 3,
    "Fast Food Restaurant": 3,
    "Grocery Store": 2,
    "Office/Indoor Workplace": 2,
    "Private Home/Residence": 2,
    "Other": 2,
    "Unknown": 1,
}
peak_months = [3,4,5,6,12]

def build_input():
    month_num = month_map[month]
    season = get_season(month_num)
    loc_risk = location_risk_map.get(location, 2)
    is_peak = 1 if month_num in peak_months else 0
    species_x_loc = f"{species_risk}_{location}"
    sor = state_rate_map.get(state, 0.5)
    fk = 1 if food_known == "Yes" else 0

    return pd.DataFrame([{
        "Year": year,
        "State": state,
        "Location": location,
        "Status": status,
        "Species_Risk": species_risk,
        "food_known": fk,
        "Food_Category": food_category,
        "Month_num": month_num,
        "Season": season,
        "location_risk": loc_risk,
        "is_peak_month": is_peak,
        "species_x_location": species_x_loc,
        "state_outbreak_rate": sor,
    }])


# ── Prediction & results ──────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        X_input = build_input()

        proba = pipeline.predict_proba(X_input)[0]

        # Binary model: class 0 = Not High, class 1 = High
        # Check which class is "High" (1 in binary target)
        prob_high = proba[1]
        prob_not_high = proba[0]
        is_high = prob_high >= threshold

        if is_high:
            st.markdown(f"""
            <div class="result-high">
                <div style="font-size:2.5rem">🔴</div>
                <div class="result-label" style="color:#ff5252">HIGH SEVERITY</div>
                <div style="color:#ffcdd2; margin-top:0.5rem; font-size:0.9rem">
                    Probability: <strong>{prob_high:.1%}</strong> · Threshold: {threshold}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-nothigh">
                <div style="font-size:2.5rem">🟢</div>
                <div class="result-label" style="color:#69f0ae">NOT HIGH</div>
                <div style="color:#c8e6c9; margin-top:0.5rem; font-size:0.9rem">
                    Probability of High: <strong>{prob_high:.1%}</strong> · Threshold: {threshold}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Probability breakdown")
        st.progress(float(prob_high), text=f"High severity: {prob_high:.1%}")
        st.progress(float(prob_not_high), text=f"Not High: {prob_not_high:.1%}")

        # ── SHAP waterfall ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">SHAP Explanation</div>', unsafe_allow_html=True)

        try:
            # Extract the preprocessor and classifier from the imblearn pipeline
            pre = pipeline.named_steps["pre"]
            clf = pipeline.named_steps["clf"]

            X_transformed = pre.transform(X_input)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(X_transformed)

            # For binary, shap_values may be 3D (1 sample, n_features, 2 classes)
            if len(shap_values.shape) == 3:
                sv = shap_values[:, :, 1]  # class 1 = High
            else:
                sv = shap_values

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#1a1d27")
            ax.set_facecolor("#1a1d27")
            shap.plots.waterfall(sv[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem; color: #6c7293;">
            <div style="font-size:3rem">🦠</div>
            <div style="font-family:'IBM Plex Mono',monospace; margin-top:1rem;">
                Configure parameters and click<br><strong>Predict Severity</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6c7293; font-size:0.8rem; font-family:IBM Plex Mono,monospace'>"
    "CDC FDOSS 1998–2015 · LightGBM_binary_final · Macro F1 69.6% · ROC-AUC 82.7%"
    "</div>",
    unsafe_allow_html=True
)