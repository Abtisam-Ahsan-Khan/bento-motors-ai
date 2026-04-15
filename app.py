import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Bento Motors AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# LOAD MODELS + DEPLOYMENT FILES
# -----------------------------
@st.cache_resource
def load_models():
    reg = joblib.load("regression_model.joblib")
    clf = joblib.load("classification_model.joblib")
    sc = joblib.load("scaler.joblib")
    with open("columns.pkl", "rb") as f:
        cols = pickle.load(f)
    with open("make_model_map.pkl", "rb") as f:
        mm_map = pickle.load(f)
    with open("model_target_encoding.pkl", "rb") as f:
        target_enc = pickle.load(f)
    return reg, clf, sc, cols, mm_map, target_enc


# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_dataset():
    possible_csv_files = [
        "adverts 1 (1).csv",
        "adverts.csv",
        "dataset.csv"
    ]

    possible_zip_files = [
        "adverts 1 (1).zip",
        "adverts.zip",
        "dataset.zip"
    ]

    for file_name in possible_csv_files:
        if Path(file_name).exists():
            return pd.read_csv(file_name)

    for zip_name in possible_zip_files:
        if Path(zip_name).exists():
            with zipfile.ZipFile(zip_name) as z:
                csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]
                if not csv_files:
                    raise FileNotFoundError(f"No CSV file found inside {zip_name}")
                with z.open(csv_files[0]) as f:
                    return pd.read_csv(f)

    raise FileNotFoundError(
        "Dataset file not found. Add adverts 1 (1).csv or adverts 1 (1).zip to the project folder."
    )


def clean_unique_values(series):
    return sorted(series.dropna().astype(str).str.strip().unique().tolist())


reg_model, class_model, scaler, model_columns, make_model_map, target_encoding = load_models()
df = load_dataset()

# Dropdown values from dataset
available_makes = sorted(make_model_map.keys())
available_fuels = clean_unique_values(df["fuel_type"]) if "fuel_type" in df.columns else []
available_bodies = clean_unique_values(df["body_type"]) if "body_type" in df.columns else []
available_colours = clean_unique_values(df["standard_colour"]) if "standard_colour" in df.columns else []

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .stApp { background: #fbfbfd; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #d2d2d7; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #86868b; }

    .hero-section {
        background: linear-gradient(180deg, #000000 0%, #1d1d1f 100%);
        border-radius: 0 0 40px 40px;
        padding: 80px 40px 100px 40px;
        margin: -1rem -1rem 60px -1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 150%;
        height: 100%;
        background: radial-gradient(ellipse at 50% 0%, rgba(142, 142, 147, 0.18) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 500;
        color: rgba(255,255,255,0.8);
        letter-spacing: 0.5px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeInDown 0.8s ease-out;
    }

    .hero-title {
        font-size: 72px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -3px;
        line-height: 1.05;
        margin: 0 0 20px 0;
        animation: fadeInUp 0.8s ease-out 0.1s both;
    }

    .hero-title span {
        background: linear-gradient(135deg, #d1d1d6 0%, #8e8e93 50%, #636366 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 24px;
        font-weight: 400;
        color: rgba(255,255,255,0.6);
        margin-bottom: 50px;
        letter-spacing: -0.3px;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }

    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 80px;
        animation: fadeInUp 0.8s ease-out 0.3s both;
        flex-wrap: wrap;
    }

    .stat-item { text-align: center; }

    .stat-number {
        font-size: 56px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -2px;
        line-height: 1;
    }

    .stat-label {
        font-size: 14px;
        font-weight: 500;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
    }

    .section-header {
        font-size: 40px;
        font-weight: 700;
        color: #1d1d1f;
        letter-spacing: -1px;
        margin-bottom: 12px;
        text-align: center;
    }

    .section-subheader {
        font-size: 18px;
        color: #86868b;
        font-weight: 400;
        text-align: center;
        margin-bottom: 40px;
    }

    .result-box {
        background: linear-gradient(135deg, #1d1d1f 0%, #2d2d2d 100%);
        padding: 50px 40px;
        border-radius: 28px;
        text-align: center;
        color: white;
        margin: 32px 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }

    .result-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 30% 0%, rgba(142,142,147,0.20) 0%, transparent 50%),
            radial-gradient(ellipse at 70% 100%, rgba(99,99,102,0.14) 0%, transparent 50%);
        pointer-events: none;
    }

    .result-box .label {
        font-size: 13px;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 12px;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    .result-box .price {
        font-size: 64px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -2px;
        position: relative;
        z-index: 1;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .cat-budget {
        background: linear-gradient(135deg, #30d158 0%, #34c759 100%);
        padding: 16px 32px;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 16px;
        font-weight: 600;
        margin-top: 16px;
        display: inline-block;
        box-shadow: 0 4px 16px rgba(48, 209, 88, 0.3);
    }

    .cat-mid {
        background: linear-gradient(135deg, #ff9f0a 0%, #ffcc00 100%);
        padding: 16px 32px;
        border-radius: 16px;
        text-align: center;
        color: #1d1d1f;
        font-size: 16px;
        font-weight: 600;
        margin-top: 16px;
        display: inline-block;
        box-shadow: 0 4px 16px rgba(255, 159, 10, 0.3);
    }

    .cat-premium {
        background: linear-gradient(135deg, #8e8e93 0%, #636366 100%);
        padding: 16px 32px;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 16px;
        font-weight: 600;
        margin-top: 16px;
        display: inline-block;
        box-shadow: 0 4px 16px rgba(99, 99, 102, 0.3);
    }

    .info-box {
        background: #ffffff;
        padding: 32px;
        border-radius: 20px;
        border: 1px solid rgba(0,0,0,0.04);
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        margin: 16px 0;
    }

    .info-box h4 {
        color: #1d1d1f;
        margin: 0 0 16px 0;
        font-size: 18px;
        font-weight: 600;
    }

    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 14px 0;
        border-bottom: 1px solid #f5f5f7;
    }

    .summary-row:last-child { border-bottom: none; }
    .summary-label { color: #86868b; font-weight: 500; font-size: 14px; }
    .summary-value { color: #1d1d1f; font-weight: 600; font-size: 14px; }

    .stButton > button {
        background: linear-gradient(135deg, #8e8e93 0%, #636366 100%);
        color: white;
        border: none;
        padding: 16px 48px;
        border-radius: 14px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
        box-shadow: 0 4px 16px rgba(99, 99, 102, 0.3);
        letter-spacing: -0.2px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99, 99, 102, 0.35);
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: #f5f5f7 !important;
        border: 2px solid transparent !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
    }

    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        background: #ebebed !important;
    }

    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        background: #ffffff !important;
        border-color: #8e8e93 !important;
        box-shadow: 0 0 0 4px rgba(142, 142, 147, 0.12) !important;
    }

    .stSelectbox label,
    .stNumberInput label {
        color: #1d1d1f !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }

    .github-link {
        text-align: center;
        margin-bottom: 40px;
    }

    .github-link a {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #636366;
        text-decoration: none;
        font-weight: 600;
        font-size: 15px;
        padding: 12px 24px;
        background: rgba(99, 99, 102, 0.08);
        border-radius: 30px;
        transition: all 0.3s ease;
    }

    .github-link a:hover {
        background: rgba(99, 99, 102, 0.15);
        transform: translateY(-2px);
    }

    .footer {
        background: #1d1d1f;
        border-radius: 28px 28px 0 0;
        padding: 50px 40px;
        margin: 80px -1rem -1rem -1rem;
        text-align: center;
    }

    .footer-brand {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .footer-text {
        color: rgba(255,255,255,0.5);
        font-size: 14px;
    }

    .footer-links {
        display: flex;
        justify-content: center;
        gap: 32px;
        margin-top: 24px;
        flex-wrap: wrap;
    }

    .footer-links a {
        color: rgba(255,255,255,0.6);
        text-decoration: none;
        font-size: 14px;
        transition: color 0.3s ease;
    }

    .footer-links a:hover { color: #ffffff; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        margin: 40px 0;
    }

    .feature-card {
        background: #ffffff;
        border-radius: 24px;
        padding: 40px 32px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
        transition: all 0.4s cubic-bezier(0.25, 0.1, 0.25, 1);
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.08);
    }

    .feature-icon {
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, #8e8e93, #636366);
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin: 0 auto 20px auto;
    }

    .feature-title {
        font-size: 20px;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 12px;
    }

    .feature-desc {
        font-size: 14px;
        color: #86868b;
        line-height: 1.6;
    }

    @media (max-width: 900px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 48px;
        }

        .hero-stats {
            gap: 32px;
        }

        .result-box .price {
            font-size: 42px;
        }
    }
</style>
""", unsafe_allow_html=True)

# HERO SECTION
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">✨ Powered by Machine Learning</div>
    <h1 class="hero-title">Bento Motors <span>AI</span></h1>
    <p class="hero-subtitle">Intelligent Vehicle Price Prediction System</p>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-number">402K+</div>
            <div class="stat-label">Vehicles Analyzed</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">68%</div>
            <div class="stat-label">R² Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">83%</div>
            <div class="stat-label">Classification</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="github-link"><a href="https://github.com/Abtisam-Ahsan-Khan/bento-motors-ai" target="_blank">📂 View Source Code on GitHub</a></div>',
    unsafe_allow_html=True
)

# FEATURE CARDS
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Accurate Predictions</div>
        <div class="feature-desc">Random Forest model trained on 400K+ vehicles with strong predictive power.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Instant Results</div>
        <div class="feature-desc">Get price predictions in seconds with a smooth business-friendly interface.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🚘</div>
        <div class="feature-title">Make-Matched Models</div>
        <div class="feature-desc">The model dropdown only shows valid models for the selected make.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# MAIN LANDING PAGE PREDICTOR
st.markdown('<p class="section-header">Predict Your Vehicle Price</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subheader">Enter your vehicle details below for an AI-powered valuation</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    make = st.selectbox("Make", available_makes)

    # use the saved make_model_map to show only valid models for the selected make
    filtered_models = make_model_map.get(make, [])
    if not filtered_models:
        filtered_models = ["No models found"]

    model_choice = st.selectbox("Model", filtered_models)
    year = st.number_input("Year of Registration", min_value=1980, max_value=2024, value=2018)

with col2:
    fuel = st.selectbox("Fuel Type", available_fuels)
    body = st.selectbox("Body Type", available_bodies)
    mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=30000, step=1000)

with col3:
    condition = st.selectbox("Condition", ["USED", "NEW"])
    colour = st.selectbox("Colour", available_colours)
    crossover = st.selectbox("Crossover Car and Van", ["No", "Yes"])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Get Price Prediction"):

    # --- BUILD THE INPUT VECTOR ---
    # start with all zeros matching the model's expected columns
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    vehicle_age = 2024 - year

    # set numerical features (raw values — will be scaled below)
    input_data["mileage"] = mileage
    input_data["year_of_registration"] = year
    input_data["vehicle_age"] = vehicle_age

    # --- TARGET ENCODE the selected model ---
    # look up the mean price for this model from the training data encoding
    model_means = target_encoding["model_means"]
    global_mean = target_encoding["global_mean"]
    input_data["standard_model_encoded"] = model_means.get(model_choice, global_mean)

    # scale ALL numerical features together (scaler expects 4 features in this order)
    num_cols = ["mileage", "year_of_registration", "vehicle_age", "standard_model_encoded"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # --- SET ONE-HOT FLAGS ---
    # condition (drop_first=True was used so only one of these exists)
    cond_col = f"vehicle_condition_{condition}"
    if cond_col in input_data.columns:
        input_data[cond_col] = 1

    # fuel, body, make, colour
    for prefix, val in [
        ("fuel_type_", fuel),
        ("body_type_", body),
        ("standard_make_", make),
        ("standard_colour_", colour),
    ]:
        col_name = f"{prefix}{val}"
        if col_name in input_data.columns:
            input_data[col_name] = 1

    # crossover
    if "crossover_car_and_van" in input_data.columns:
        input_data["crossover_car_and_van"] = 1 if crossover == "Yes" else 0

    # --- PREDICT ---
    price_pred = reg_model.predict(input_data)[0]

    # clamp to sensible range
    price_pred = max(500, price_pred)

    if price_pred < 10000:
        category, cat_class, cat_icon = "Budget", "cat-budget", "🟢"
    elif price_pred < 25000:
        category, cat_class, cat_icon = "Mid-Range", "cat-mid", "🟡"
    else:
        category, cat_class, cat_icon = "Premium", "cat-premium", "⚪"

    st.markdown(f"""
    <div class="result-box">
        <div class="label">Estimated Vehicle Price</div>
        <div class="price">£{price_pred:,.2f}</div>
    </div>
    <div style="text-align: center;">
        <div class="{cat_class}">{cat_icon} {category} Category</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header" style="font-size: 28px;">Vehicle Summary</p>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)

    with s1:
        st.markdown(f"""
        <div class="info-box">
            <h4>🚗 Vehicle Details</h4>
            <div class="summary-row"><span class="summary-label">Make</span><span class="summary-value">{make}</span></div>
            <div class="summary-row"><span class="summary-label">Model</span><span class="summary-value">{model_choice}</span></div>
            <div class="summary-row"><span class="summary-label">Body Type</span><span class="summary-value">{body}</span></div>
            <div class="summary-row"><span class="summary-label">Fuel Type</span><span class="summary-value">{fuel}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="info-box">
            <h4>📋 Specifications</h4>
            <div class="summary-row"><span class="summary-label">Mileage</span><span class="summary-value">{mileage:,} miles</span></div>
            <div class="summary-row"><span class="summary-label">Year</span><span class="summary-value">{year}</span></div>
            <div class="summary-row"><span class="summary-label">Vehicle Age</span><span class="summary-value">{vehicle_age} years</span></div>
            <div class="summary-row"><span class="summary-label">Condition</span><span class="summary-value">{condition}</span></div>
        </div>
        """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    <div class="footer-brand">🚗 Bento Motors AI</div>
    <div class="footer-text">Intelligent Vehicle Price Prediction System</div>
    <div class="footer-links">
        <a href="#">Privacy Policy</a>
        <a href="#">Terms of Service</a>
        <a href="#">Contact</a>
        <a href="https://github.com/Abtisam-Ahsan-Khan/bento-motors-ai" target="_blank">GitHub</a>
    </div>
    <div class="footer-text" style="margin-top: 24px; opacity: 0.5;">© 2026 Bento Motors AI · UA92-333 Applied AI · Built with Streamlit</div>
</div>
""", unsafe_allow_html=True)
