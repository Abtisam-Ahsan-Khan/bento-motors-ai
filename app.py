import streamlit as st
import joblib
import pickle
import pandas as pd

# Page config
st.set_page_config(
    page_title="Bento Motors AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load models and columns
@st.cache_resource
def load_models():
    reg = joblib.load("regression_model.joblib")
    clf = joblib.load("classification_model.joblib")
    sc = joblib.load("scaler.joblib")
    with open("columns.pkl", "rb") as f:
        cols = pickle.load(f)
    return reg, clf, sc, cols


reg_model, class_model, scaler, model_columns = load_models()


def extract_values_from_columns(columns, prefix):
    values = []
    for col in columns:
        if col.startswith(prefix):
            values.append(col.replace(prefix, ""))
    return sorted(values)


# Extract dropdown values from trained columns
available_makes = extract_values_from_columns(model_columns, "standard_make_")
available_models = extract_values_from_columns(model_columns, "standard_model_")
available_fuels = extract_values_from_columns(model_columns, "fuel_type_")
available_bodies = extract_values_from_columns(model_columns, "body_type_")
available_colours = extract_values_from_columns(model_columns, "standard_colour_")

# Fallbacks
if not available_makes:
    available_makes = [
        "BMW", "Audi", "Volkswagen", "Vauxhall", "Mercedes-Benz", "Nissan",
        "Toyota", "Peugeot", "Land Rover", "Renault", "Ford", "Hyundai",
        "Kia", "MINI", "Volvo", "Honda", "Citroen", "SEAT", "Mazda",
        "Jaguar", "Tesla", "Porsche", "Lexus", "Other"
    ]

if not available_models:
    available_models = [
        "A3", "A4", "A6", "X1", "X3", "X5", "C Class", "E Class",
        "Golf", "Polo", "Focus", "Fiesta", "Qashqai", "Tiguan", "Other"
    ]

if not available_fuels:
    available_fuels = [
        "Petrol", "Diesel", "Electric", "Petrol Hybrid",
        "Petrol Plug-in Hybrid", "Diesel Hybrid"
    ]

if not available_bodies:
    available_bodies = [
        "Hatchback", "SUV", "Saloon", "Estate", "Coupe",
        "Convertible", "MPV", "Pickup"
    ]

if not available_colours:
    available_colours = [
        "Black", "White", "Grey", "Blue", "Silver", "Red",
        "Green", "Orange", "Yellow", "Brown", "Other"
    ]


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
        background: radial-gradient(ellipse at 50% 0%, rgba(0, 113, 227, 0.15) 0%, transparent 60%);
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
        background: linear-gradient(135deg, #7dc3ff 0%, #4da3ff 50%, #0071e3 100%);
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
            radial-gradient(ellipse at 30% 0%, rgba(0,113,227,0.20) 0%, transparent 50%),
            radial-gradient(ellipse at 70% 100%, rgba(125,125,130,0.12) 0%, transparent 50%);
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
        background: linear-gradient(135deg, #0071e3 0%, #5ac8fa 100%);
        padding: 16px 32px;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 16px;
        font-weight: 600;
        margin-top: 16px;
        display: inline-block;
        box-shadow: 0 4px 16px rgba(0, 113, 227, 0.3);
    }

    .metric-card {
        background: #ffffff;
        padding: 32px 24px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
        transition: all 0.4s cubic-bezier(0.25, 0.1, 0.25, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0071e3, #5ac8fa);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.08);
    }

    .metric-card .value {
        color: #1d1d1f;
        margin: 0;
        font-size: 44px;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .metric-card .desc {
        color: #86868b;
        margin: 8px 0 0 0;
        font-size: 14px;
        font-weight: 500;
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

    .info-box p {
        color: #424245;
        line-height: 1.8;
        margin-bottom: 0;
        font-size: 15px;
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

    .step-card {
        background: #ffffff;
        padding: 28px 28px 28px 80px;
        border-radius: 20px;
        border: 1px solid rgba(0,0,0,0.04);
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        margin: 16px 0;
        position: relative;
        transition: all 0.3s ease;
    }

    .step-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }

    .step-num {
        position: absolute;
        left: 24px;
        top: 50%;
        transform: translateY(-50%);
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #0071e3, #5ac8fa);
        border-radius: 14px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3);
    }

    .step-card h4 {
        color: #1d1d1f;
        margin: 0 0 8px 0;
        font-size: 17px;
        font-weight: 600;
    }

    .step-card p {
        color: #86868b;
        margin: 0;
        line-height: 1.6;
        font-size: 14px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0071e3 0%, #5ac8fa 100%);
        color: white;
        border: none;
        padding: 16px 48px;
        border-radius: 14px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
        box-shadow: 0 4px 16px rgba(0, 113, 227, 0.3);
        letter-spacing: -0.2px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0, 113, 227, 0.4);
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
        border-color: #0071e3 !important;
        box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1) !important;
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
        color: #0071e3;
        text-decoration: none;
        font-weight: 600;
        font-size: 15px;
        padding: 12px 24px;
        background: rgba(0, 113, 227, 0.08);
        border-radius: 30px;
        transition: all 0.3s ease;
    }

    .github-link a:hover {
        background: rgba(0, 113, 227, 0.15);
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

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #d2d2d7, transparent);
        margin: 48px 0;
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
        background: linear-gradient(135deg, #0071e3, #5ac8fa);
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
        <div class="feature-title">Model-Level Inputs</div>
        <div class="feature-desc">Select make, model, fuel type, body type, and more for better valuations.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# MAIN LANDING PAGE PREDICTOR
st.markdown('<p class="section-header">Predict Your Vehicle Price</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subheader">Enter your vehicle details below for an AI-powered valuation</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    make = st.selectbox("Make", available_makes)
    model = st.selectbox("Model", available_models)
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

if st.button("🚀 Get Price Prediction"):
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    vehicle_age = 2024 - year

    if "mileage" in input_data.columns:
        input_data["mileage"] = mileage
    if "year_of_registration" in input_data.columns:
        input_data["year_of_registration"] = year
    if "vehicle_age" in input_data.columns:
        input_data["vehicle_age"] = vehicle_age

    numeric_features = ["mileage", "year_of_registration", "vehicle_age"]
    existing_numeric_features = [col for col in numeric_features if col in input_data.columns]

    if len(existing_numeric_features) == len(numeric_features):
        input_data[existing_numeric_features] = scaler.transform(input_data[existing_numeric_features])

    if condition == "NEW" and "vehicle_condition_NEW" in input_data.columns:
        input_data["vehicle_condition_NEW"] = 1

    for prefix, val in [
        ("fuel_type_", fuel),
        ("body_type_", body),
        ("standard_make_", make),
        ("standard_model_", model),
        ("standard_colour_", colour),
    ]:
        col_name = f"{prefix}{val}"
        if col_name in input_data.columns:
            input_data[col_name] = 1

    if crossover == "Yes" and "crossover_car_and_van" in input_data.columns:
        input_data["crossover_car_and_van"] = 1

    price_pred = reg_model.predict(input_data)[0]

    if price_pred < 10000:
        category, cat_class, cat_icon = "Budget", "cat-budget", "🟢"
    elif price_pred < 25000:
        category, cat_class, cat_icon = "Mid-Range", "cat-mid", "🟡"
    else:
        category, cat_class, cat_icon = "Premium", "cat-premium", "🔵"

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
            <div class="summary-row"><span class="summary-label">Model</span><span class="summary-value">{model}</span></div>
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
