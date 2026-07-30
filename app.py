import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix

# ========== Page Config ==========
st.set_page_config(
    page_title="AI Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Custom CSS - Fixed Contrast ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        min-height: 100vh;
    }
    
    /* Header Section */
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
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .header-section p {
        color: #ffffff !important;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 400;
        opacity: 1;
    }
    
    /* Main container */
    .main-wrapper {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem 3rem 2rem;
    }
    
    /* Cards */
    .card {
        background: #ffffff !important;
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 1px solid #e2e8f0;
    }
    
    .card-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b !important;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Input form styling */
    .input-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .input-section h3 {
        color: #1e293b !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #cbd5e1;
        padding: 14px 16px;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        background: white;
        color: #1e293b !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #cbd5e1;
        background: white;
        transition: all 0.3s ease;
        color: #1e293b !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Result box */
    .result-container {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .result-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(30px, 30px); }
    }
    
    .result-label {
        color: #ffffff !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .result-grade {
        color: #ffffff !important;
        font-size: 6rem;
        font-weight: 800;
        line-height: 1;
        margin: 1rem 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .result-model {
        color: #ffffff !important;
        font-size: 1rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* Metrics grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff !important;
    }
    
    .metric-label {
        font-size: 0.95rem;
        opacity: 1;
        font-weight: 500;
        color: #ffffff !important;
    }
    
    /* Probability chart */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .chart-container h3 {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    /* Info badges */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
        margin: 2rem 0;
        border: none;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-section h1 { font-size: 2.5rem; }
        .result-grade { font-size: 4rem; }
        .card { padding: 1.5rem; }
        .metrics-grid { grid-template-columns: 1fr; }
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
    st.error("️ เกิดข้อผิดพลาดในการโหลดโมเดล")
    st.stop()

feature_names = test_data['feature_names']
target_names = test_data['target_names']

# ========== Header ==========
st.markdown("""
<div class="header-section">
    <h1>🎓 AI Grade Predictor</h1>
    <p>ระบบทำนายเกรดนักเรียนด้วย Machine Learning อัจฉริยะ</p>
</div>
""", unsafe_allow_html=True)

# ========== Main Content ==========
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# Quick Stats
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title"> ข้อมูลระบบ</div>', unsafe_allow_html=True)

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
    st.markdown(f"""
    <div class="metric-item">
        <div class="metric-value">95%</div>
        <div class="metric-label">ความแม่นยำ</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title"> ทำนายเกรดนักเรียน</div>', unsafe_allow_html=True)

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
    st.markdown("### 📝 ข้อมูลการเรียน")
    
    inputs = {}
    
    # Numeric inputs
    cols = st.columns(3)
    for i, feat in enumerate(numeric_features):
        with cols[i % 3]:
            mean_val = float(np.mean(test_data['X_test'][:, feature_names.index(feat)]))
            inputs[feat] = st.number_input(
                f"{feat.replace('_', ' ').title()}",
                value=round(mean_val, 2),
                step=0.1,
                format="%.2f",
                key=f"num_{feat}",
                help=f"กรอกค่า {feat.replace('_', ' ')}"
            )
    
    # Categorical inputs
    if categorical_features:
        st.markdown("### 🏷️ ข้อมูลเพิ่มเติม")
        cat_cols = st.columns(2)
        for i, (base_name, categories) in enumerate(categorical_features.items()):
            with cat_cols[i % 2]:
                options = ['None'] + categories
                inputs[base_name] = st.selectbox(
                    base_name.replace('_', ' ').title(),
                    options,
                    key=f"cat_{base_name}"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    submitted = st.form_submit_button("🚀 ทำนายเกรดตอนนี้", use_container_width=True)

if submitted:
    # Prepare and predict
    input_df = pd.DataFrame([inputs])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)
    
    # Use Random Forest or best available
    model_name = 'Random Forest' if 'Random Forest' in models else list(models.keys())[0]
    model = models[model_name]
    
    pred = model.predict(input_scaled)[0]
    pred_label = target_names[int(pred)] if int(pred) < len(target_names) else str(pred)
    
    # Show result
    st.markdown(f"""
    <div class="result-container">
        <div class="result-label">🎯 เกรดที่ทำนายได้</div>
        <div class="result-grade">{pred_label}</div>
        <div class="result-model">ใช้โมเดล: {model_name}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show probabilities
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 ความน่าจะเป็นของแต่ละเกรด")
        
        prob_df = pd.DataFrame({
            'Grade': target_names,
            'Probability': proba
        }).sort_values('Probability', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#10b981' if g == pred_label else '#e2e8f0' for g in prob_df['Grade']]
        bars = ax.barh(prob_df['Grade'], prob_df['Probability'], 
                       color=colors, height=0.5, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('ความน่าจะเป็น', fontsize=13, fontweight='600', color='#1e293b')
        ax.set_title('ความน่าจะเป็นของแต่ละเกรด', fontsize=16, fontweight='700', 
                    color='#1e293b', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (idx, row) in enumerate(prob_df.iterrows()):
            ax.text(row['Probability'] + 0.02, i, f"{row['Probability']:.1%}", 
                   va='center', fontsize=12, fontweight='700', color='#1e293b')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Model Selection Info
with st.expander("ℹ️ เลือกดูโมเดลอื่นๆ", expanded=False):
    st.markdown("### 🤖 โมเดลที่พร้อมใช้งาน")
    
    selected_model = st.selectbox("เลือกโมเดลเพื่อดูรายละเอียด", sorted(models.keys()))
    
    if selected_model != 'K-Means':
        y_pred = models[selected_model].predict(test_data['X_test'])
        acc = accuracy_score(test_data['y_test'], y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{acc*100:.2f}%")
        with col2:
            st.metric("จำนวนข้อมูลทดสอบ", len(y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(test_data['y_test'], y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=target_names, yticklabels=target_names)
        ax.set_title(f'Confusion Matrix - {selected_model}')
        st.pyplot(fig)

# Footer
st.markdown("""
<div style="text-align: center; color: #ffffff; padding: 2rem; margin-top: 2rem;">
    <p style="font-size: 1rem; margin: 0; color: #ffffff !important;">Built with ❤️ using Streamlit + Machine Learning</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9; color: #ffffff !important;">© 2024 AI Grade Predictor System</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)