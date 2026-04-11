import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd

# ─── Load Models & Scalers ───
@st.cache_resource
def load_models():
    reg = joblib.load('regression_model.joblib')
    clf = joblib.load('classification_model.joblib')
    sc = joblib.load('scaler.joblib')
    with open('columns.pkl', 'rb') as f:
        cols = pickle.load(f)
    return reg, clf, sc, cols

reg_model, class_model, scaler, model_columns = load_models()

st.set_page_config(page_title="Bento Motors AI", page_icon="🚗", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Sora:wght@300;400;500;600;700&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(180deg, #ffffff 0%, #ffffff 40%, #dceefb 60%, #5ba3d9 80%, #1a3d5c 100%); background-attachment: fixed; min-height: 100vh; }
    .hero { text-align: center; padding: 40px 20px 20px 20px; }
    .hero h1 { font-family: 'Sora', sans-serif; font-size: 3em; font-weight: 700; color: #1a3d5c; margin: 0; letter-spacing: -1px; }
    .hero .subtitle { font-size: 1.15em; color: #5a7a94; margin-top: 6px; font-weight: 400; }
    .hero .divider { width: 60px; height: 4px; background: linear-gradient(90deg, #2980b9, #1a3d5c); margin: 15px auto; border-radius: 2px; }
    .github-link { text-align: center; margin-bottom: 15px; }
    .github-link a { color: #2980b9; text-decoration: none; font-weight: 500; font-size: 0.95em; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; justify-content: center; background: rgba(255,255,255,0.7); backdrop-filter: blur(10px); border-radius: 14px; padding: 6px; max-width: 700px; margin: 0 auto 25px auto; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 10px; padding: 10px 22px; color: #1a3d5c; font-weight: 600; font-size: 0.92em; border: none; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #2980b9, #1a3d5c) !important; color: white !important; box-shadow: 0 4px 12px rgba(26, 61, 92, 0.3); }
    .result-box { background: linear-gradient(135deg, #1a3d5c, #2980b9); padding: 35px; border-radius: 18px; text-align: center; color: white; margin: 25px 0; box-shadow: 0 8px 30px rgba(26, 61, 92, 0.35); }
    .result-box .label { font-size: 0.95em; opacity: 0.85; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; font-weight: 500; }
    .result-box .price { font-family: 'Sora', sans-serif; font-size: 2.8em; font-weight: 700; margin: 0; }
    .cat-budget { background: linear-gradient(135deg, #1e8449, #27ae60); padding: 14px; border-radius: 12px; text-align: center; color: white; font-size: 1.15em; font-weight: 600; margin-top: 12px; }
    .cat-mid { background: linear-gradient(135deg, #d68910, #f39c12); padding: 14px; border-radius: 12px; text-align: center; color: white; font-size: 1.15em; font-weight: 600; margin-top: 12px; }
    .cat-premium { background: linear-gradient(135deg, #c0392b, #e74c3c); padding: 14px; border-radius: 12px; text-align: center; color: white; font-size: 1.15em; font-weight: 600; margin-top: 12px; }
    .metric-card { background: white; padding: 25px 20px; border-radius: 14px; text-align: center; border-left: 5px solid #2980b9; box-shadow: 0 3px 12px rgba(0,0,0,0.08); margin: 8px 0; }
    .metric-card .value { font-family: 'Sora', sans-serif; color: #1a3d5c; margin: 0; font-size: 2em; font-weight: 700; }
    .metric-card .desc { color: #6b8299; margin: 6px 0 0 0; font-size: 0.9em; font-weight: 500; }
    .info-box { background: white; padding: 22px 25px; border-radius: 14px; border: 1px solid #e8eff5; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 12px 0; }
    .info-box h4 { color: #1a3d5c; font-family: 'Sora', sans-serif; margin-top: 0; font-size: 1.1em; }
    .info-box p { color: #4a6a80; line-height: 1.7; margin-bottom: 0; }
    .step-card { background: white; padding: 25px; border-radius: 14px; border: 1px solid #e8eff5; margin: 12px 0; position: relative; padding-left: 70px; }
    .step-num { position: absolute; left: 18px; top: 50%; transform: translateY(-50%); width: 38px; height: 38px; background: linear-gradient(135deg, #2980b9, #1a3d5c); border-radius: 10px; color: white; display: flex; align-items: center; justify-content: center; font-family: 'Sora', sans-serif; font-weight: 700; }
    .step-card h4 { color: #1a3d5c; font-family: 'Sora', sans-serif; margin: 0 0 6px 0; }
    .step-card p { color: #4a6a80; margin: 0; line-height: 1.6; }
    .summary-row { display: flex; justify-content: space-between; padding: 10px 15px; border-bottom: 1px solid #eef3f8; }
    .summary-row:last-child { border-bottom: none; }
    .summary-label { color: #6b8299; font-weight: 500; }
    .summary-value { color: #1a3d5c; font-weight: 600; }
    .stButton > button { background: linear-gradient(135deg, #2980b9, #1a3d5c); color: white; border: none; padding: 14px 40px; border-radius: 12px; font-size: 1.05em; font-weight: 600; width: 100%; box-shadow: 0 4px 15px rgba(26, 61, 92, 0.25); }
    .section-header { font-family: 'Sora', sans-serif; color: #1a3d5c; font-size: 1.5em; font-weight: 600; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 3px solid #2980b9; display: inline-block; }
    .footer { text-align: center; color: rgba(255,255,255,0.7); padding: 30px 20px; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🚗 Bento Motors AI</h1><div class="divider"></div><p class="subtitle">Intelligent Vehicle Price Prediction System</p></div>', unsafe_allow_html=True)
st.markdown('<div class="github-link"><a href="https://github.com/Abtisam-Ahsan-Khan/bento-motors-ai" target="_blank">📂 View Source Code on GitHub</a></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Price Predictor", "📊 Model Performance", "🧠 How It Works", "ℹ️ About"])

with tab1:
    st.markdown('<div class="section-header">Enter Vehicle Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        mileage = st.number_input("Mileage", min_value=0, max_value=200000, value=30000, step=1000)
        year = st.number_input("Year of Registration", min_value=1980, max_value=2024, value=2018)
        condition = st.selectbox("Condition", ["USED", "NEW"])
    with col2:
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Petrol Hybrid", "Petrol Plug-in Hybrid", "Diesel Hybrid"])
        body = st.selectbox("Body Type", ["Hatchback", "SUV", "Saloon", "Estate", "Coupe", "Convertible", "MPV", "Pickup"])
        crossover = st.selectbox("Crossover Car and Van", ["No", "Yes"])
    with col3:
        make = st.selectbox("Make", ["BMW", "Audi", "Volkswagen", "Vauxhall", "Mercedes-Benz", "Nissan", "Toyota", "Peugeot", "Land Rover", "Renault", "Ford", "Hyundai", "Kia", "MINI", "Volvo", "Honda", "Citroen", "SEAT", "Mazda", "Jaguar", "Other"])
        colour = st.selectbox("Colour", ["Black", "White", "Grey", "Blue", "Silver", "Red", "Green", "Orange", "Yellow", "Brown", "Other"])
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Predict Price"):
        input_data = pd.DataFrame(0, index=[0], columns=model_columns)
        vehicle_age = 2024 - year
        input_data['mileage'] = mileage
        input_data['year_of_registration'] = year
        input_data['vehicle_age'] = vehicle_age
        num_features = ['mileage', 'year_of_registration', 'vehicle_age']
        input_data[num_features] = scaler.transform(input_data[num_features])
        if condition == "NEW" and 'vehicle_condition_NEW' in input_data.columns:
            input_data['vehicle_condition_NEW'] = 1
        for prefix, val in [('fuel_type_', fuel), ('body_type_', body), ('standard_make_', make), ('standard_colour_', colour)]:
            col_name = f'{prefix}{val}'
            if col_name in input_data.columns:
                input_data[col_name] = 1
        if crossover == "Yes":
            input_data['crossover_car_and_van'] = 1
        price_pred = reg_model.predict(input_data)[0]
        if price_pred < 10000:
            category, cat_class, cat_icon = "Budget", "cat-budget", "🟢"
        elif price_pred < 25000:
            category, cat_class, cat_icon = "Mid-Range", "cat-mid", "🟡"
        else:
            category, cat_class, cat_icon = "Premium", "cat-premium", "🔴"
        st.markdown(f'<div class="result-box"><div class="label">ESTIMATED VEHICLE PRICE</div><div class="price">£{price_pred:,.2f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{cat_class}">{cat_icon} Price Category: {category}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Vehicle Summary</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f'<div class="info-box"><div class="summary-row"><span class="summary-label">Make</span><span class="summary-value">{make}</span></div><div class="summary-row"><span class="summary-label">Body Type</span><span class="summary-value">{body}</span></div><div class="summary-row"><span class="summary-label">Fuel Type</span><span class="summary-value">{fuel}</span></div><div class="summary-row"><span class="summary-label">Colour</span><span class="summary-value">{colour}</span></div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="info-box"><div class="summary-row"><span class="summary-label">Mileage</span><span class="summary-value">{mileage:,} miles</span></div><div class="summary-row"><span class="summary-label">Year</span><span class="summary-value">{year}</span></div><div class="summary-row"><span class="summary-label">Vehicle Age</span><span class="summary-value">{vehicle_age} years</span></div><div class="summary-row"><span class="summary-label">Condition</span><span class="summary-value">{condition}</span></div></div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-header">Regression Model — Random Forest</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><div class="value">0.68</div><div class="desc">Test R² Score</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><div class="value">£9,284</div><div class="desc">Test RMSE</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><div class="value">68%</div><div class="desc">Variance Explained</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><p>The regression model predicts exact vehicle prices using a <strong>Tuned Random Forest Regressor</strong>. It was optimised through two rounds of GridSearchCV. The model explains 68% of price variation — the remaining 32% is due to features not in the dataset such as engine size and service history.</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">Classification Model — Decision Tree</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="value">83%</div><div class="desc">Test Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="value">3</div><div class="desc">Categories</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="value">2 Rounds</div><div class="desc">Tuning Iterations</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><p>The classification model groups vehicles into <strong>Budget</strong> (under £10k), <strong>Mid-Range</strong> (£10k-£25k), and <strong>Premium</strong> (over £25k). This helps Bento Motors quickly classify incoming stock.</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">Top Predictors</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><p>📉 <strong>Mileage</strong> — higher mileage reduces value</p><br><p>📅 <strong>Vehicle Age</strong> — steepest drop in first 3-4 years</p><br><p>🏭 <strong>Make</strong> — premium brands command higher prices</p><br><p>⛽ <strong>Fuel Type</strong> — electric/hybrid valued higher</p><br><p>🚗 <strong>Body Type</strong> — SUVs and convertibles attract premiums</p></div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header">The ML Pipeline</div>', unsafe_allow_html=True)
    for num, title, desc in [("1","Data Collection","Analysed over 400,000 vehicle advertisements from Bento Motors."),("2","Data Processing","Cleaned errors, handled missing data, removed outliers, encoded categories, scaled features."),("3","Model Training","Tested Linear Regression, Decision Tree, Random Forest for regression and Decision Tree, Logistic Regression for classification."),("4","Hyperparameter Tuning","GridSearchCV across two rounds, retraining best params on full data."),("5","SHAP Interpretation","Applied SHAP for transparency — explaining which features drive each prediction."),("6","Deployment","Saved models with Joblib and deployed this Streamlit app.")]:
        st.markdown(f'<div class="step-card"><div class="step-num">{num}</div><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><h4>🚗 Bento Motors AI Price Predictor</h4><p>Developed for <strong>Applied AI (UA92-333)</strong> at University Academy 92, Manchester.</p><br><div class="summary-row"><span class="summary-label">Dataset</span><span class="summary-value">402,005 vehicle ads</span></div><div class="summary-row"><span class="summary-label">Regression</span><span class="summary-value">Tuned Random Forest (R² = 0.68)</span></div><div class="summary-row"><span class="summary-label">Classification</span><span class="summary-value">Tuned Decision Tree (83%)</span></div><div class="summary-row"><span class="summary-label">Interpretation</span><span class="summary-value">SHAP Framework</span></div><div class="summary-row"><span class="summary-label">Deployment</span><span class="summary-value">Streamlit Cloud</span></div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">Ethical AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><p>🔹 Model uses objective features (mileage, age, condition) — fair and transparent</p><br><p>🔹 Colour has minor influence — should be monitored for bias</p><br><p>🔹 Under <strong>GDPR Article 22</strong>, customers can request explanation of automated decisions</p><br><p>🔹 <strong>EU AI Act</strong> classifies this as limited-risk — transparency obligations apply</p><br><p>🔹 Regular audits recommended to detect bias or drift</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">Limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><p>🔹 Missing: engine size, service history, previous owners, trim level</p><br><p>🔹 Based on historical data — may not reflect current market</p><br><p>🔹 68% variance explained — 32% from unmeasured factors</p><br><p>🔹 Rare/luxury vehicles may get less accurate predictions</p></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="footer">Bento Motors AI · © 2026 · UA92-333 Applied AI · Built with Streamlit</div>', unsafe_allow_html=True)
