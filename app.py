import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
from collections import Counter

# ========== Page Config ==========
st.set_page_config(
    page_title="AI Grade Predictor - All Models",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Custom CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', sans-serif; box-sizing: border-box; }
    
    #MainMenu, footer, header { visibility: hidden !important; }
    
    .stApp { background: #0f172a !important; min-height: 100vh; }
    
    /* Header */
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 0 0 40px 40px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-section h1 {
        color: #ffffff !important;
        font-size: 3.2rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header-section p {
        color: #ffffff !important;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Main wrapper */
    .main-wrapper { max-width: 1400px; margin: 0 auto; padding: 0 2rem 3rem 2rem; }
    
    /* Cards */
    .card {
        background: #ffffff !important;
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 2px solid #e2e8f0;
    }
    
    .section-title-dark {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a !important;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 4px solid #667eea;
    }
    
    /* Metric items */
    .metric-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff !important;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
        height: 100%;
    }
    .metric-item:hover { transform: translateY(-5px); }
    .metric-value { font-size: 3rem; font-weight: 900; color: #ffffff !important; margin-bottom: 0.5rem; }
    .metric-label { font-size: 1rem; font-weight: 600; color: #ffffff !important; }
    
    /* Input section */
    .input-section {
        background: #f1f5f9 !important;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 2px solid #cbd5e1;
    }
    .input-section h3 { color: #0f172a !important; font-weight: 700; font-size: 1.3rem; margin-bottom: 1rem; }
    
    .stNumberInput label { color: #0f172a !important; font-weight: 600; font-size: 1rem; }
    .stNumberInput input[type="number"] {
        background: #ffffff !important;
        color: #0f172a !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border: 2px solid #94a3b8 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }
    .stSelectbox label { color: #0f172a !important; font-weight: 600; font-size: 1rem; }
    .stSelectbox > div > div {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 2px solid #94a3b8 !important;
        border-radius: 12px !important;
    }
    .stSelectbox > div > div > div { color: #0f172a !important; font-weight: 600; }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none;
        border-radius: 16px;
        padding: 1.2rem 3rem;
        font-size: 1.3rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Result cards */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .model-result-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 20px;
        padding: 2rem;
        border-left: 6px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .model-result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .model-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 0.5rem;
    }
    
    .model-prediction {
        font-size: 3rem;
        font-weight: 900;
        color: #667eea !important;
        margin: 1rem 0;
        text-align: center;
    }
    
    .model-confidence {
        font-size: 0.95rem;
        font-weight: 600;
        color: #475569 !important;
        text-align: center;
    }
    
    .model-accuracy {
        background: #667eea;
        color: #ffffff !important;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Final result box */
    .final-result {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.4);
    }
    .final-result-label { color: #ffffff !important; font-size: 1.2rem; font-weight: 600; }
    .final-result-grade {
        color: #ffffff !important;
        font-size: 7rem;
        font-weight: 900;
        line-height: 1;
        margin: 1rem 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    .final-result-detail { color: #ffffff !important; font-size: 1.1rem; font-weight: 500; }
    
    /* Comparison chart */
    .chart-container {
        background: #ffffff !important;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        border: 2px solid #e2e8f0;
    }
    
    details {
        background: #ffffff !important;
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        border: 2px solid #e2e8f0;
    }
    summary { color: #0f172a !important; font-weight: 700; font-size: 1.1rem; }
    
    @media (max-width: 768px) {
        .header-section h1 { font-size: 2.2rem; }
        .final-result-grade { font-size: 4rem; }
        .card { padding: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ========== Load Models ==========
@st.cache_resource
def load_all():
    models_path = 'models'
    if not os.path.exists(models_path):
        return None, None, None
    
    models = {}
    for f in os.listdir(models_path):
        if f.endswith('.pkl') and f not in ['scaler.pkl', 'test_data.pkl']:
            name = f.replace('_', ' ').replace('.pkl', '')
            models[name] = joblib.load(os.path.join(models_path, f))
    
    scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
    test_data = joblib.load(os.path.join(models_path, 'test_data.pkl'))
    
    return models, scaler, test_data

models, scaler, test_data = load_all()

if models is None or not models or test_data is None:
    st.error("⚠️ เกิดข้อผิดพลาดในการโหลดโมเดล")
    st.stop()

feature_names = test_data['feature_names']
target_names = test_data['target_names']

# คำนวณ accuracy ของแต่ละโมเดล
model_accuracies = {}
for name, model in models.items():
    if name != 'K-Means':
        y_pred = model.predict(test_data['X_test'])
        model_accuracies[name] = accuracy_score(test_data['y_test'], y_pred)

# ========== Header ==========
st.markdown("""
<div class="header-section">
    <h1>🎓 AI Grade Predictor</h1>
    <p>ระบบทำนายเกรดนักเรียนด้วย Machine Learning - เปรียบเทียบทุกโมเดล</p>
</div>
""", unsafe_allow_html=True)

# ========== Main Content ==========
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# ========== Quick Stats ==========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title-dark"> ข้อมูลระบบ</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-item">
        <div class="metric-value">{len(models)}</div>
        <div class="metric-label">โมเดล AI</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-item">
        <div class="metric-value">{len(feature_names)}</div>
        <div class="metric-label">Features</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-item">
        <div class="metric-value">{len(target_names)}</div>
        <div class="metric-label">เกรด</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    best_acc = max(model_accuracies.values()) if model_accuracies else 0
    st.markdown(f"""
    <div class="metric-item">
        <div class="metric-value">{best_acc*100:.0f}%</div>
        <div class="metric-label">ความแม่นยำสูงสุด</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ========== Prediction Form ==========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title-dark">🔮 กรอกข้อมูลเพื่อทำนาย</div>', unsafe_allow_html=True)

# แยก features
numeric_features = [f for f in feature_names if '_' not in f]
categorical_features = {}
for feat in feature_names:
    if '_' in feat:
        base, val = feat.rsplit('_', 1)
        if base not in categorical_features:
            categorical_features[base] = []
        categorical_features[base].append(val)

with st.form("prediction_form", clear_on_submit=False):
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("###  ข้อมูลการเรียน")
    
    inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(numeric_features):
        with cols[i % 2]:
            mean_val = float(np.mean(test_data['X_test'][:, feature_names.index(feat)]))
            default_val = round(mean_val, 1)
            inputs[feat] = st.number_input(
                f"{feat.replace('_', ' ').title()}",
                value=default_val,
                step=0.5,
                format="%.1f",
                key=f"num_{feat}"
            )
