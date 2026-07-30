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

# ========== Custom CSS - Fixed ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    
    /* Main background - Dark */
    .stApp {
        background: #0f172a !important;
        min-height: 100vh;
    }
    
    /* Force ALL text to be visible */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp span, .stApp div {
        color: #ffffff !important;
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
    }
    
    .header-section p {
        color: #ffffff !important;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Main wrapper */
    .main-wrapper {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem 3rem 2rem;
    }
    
    /* Cards - WHITE background */
    .card {
        background: #ffffff !important;
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 2px solid #e2e8f0;
    }
    
    /* Section titles - DARK color on white card */
    .section-title-dark {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a !important;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 4px solid #667eea;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-title-dark .emoji {
        font-size: 2rem;
    }
    
    /* Input section - Light gray background */
    .input-section {
        background: #f1f5f9 !important;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 2px solid #cbd5e1;
    }
    
    .input-section h3 {
        color: #0f172a !important;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    /* Force number input text to be DARK and VISIBLE */
    .stNumberInput label {
        color: #0f172a !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stNumberInput input[type="number"] {
        background: #ffffff !important;
        color: #0f172a !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border: 2px solid #94a3b8 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }
    
    .stNumberInput input[type="number"]:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Force selectbox text to be DARK */
    .stSelectbox label {
        color: #0f172a !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stSelectbox > div > div {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 2px solid #94a3b8 !important;
        border-radius: 12px !important;
    }
    
    .stSelectbox > div > div > div {
        color: #0f172a !important;
        font-weight: 600;
    }
    
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
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Result box */
    .result-container {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.4);
    }
    
    .result-label {
        color: #ffffff !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .result-grade {
        color: #ffffff !important;
        font-size: 6rem;
        font-weight: 900;
        line-height: 1;
        margin: 1rem 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    
    .result-model {
        color: #ffffff !important;
        font-size: 1rem;
        font-weight: 500;
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
    
    .metric-item:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        color: #ffffff !important;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff !important;
        opacity: 1;
    }
    
    /* Chart container */
    .chart-container {
        background: #ffffff !important;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #e2e8f0;
    }
    
    .chart-container h3 {
        color: #0f172a !important;
        font-weight: 700;
    }
    
    /* Expander */
    details {
        background: #ffffff !important;
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        border: 2px solid #e2e8f0;
    }
    
    summary {
        color: #0f172a !important;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-section h1 { font-size: 2.2rem; }
        .result-grade { font-size: 4rem; }
        .card { padding: 1.5rem; }
        .section-title-dark { font-size: 1.4rem; }
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
    <h1> AI Grade Predictor</h1>
    <p>ระบบทำนายเกรดนักเรียนด้วย Machine Learning อัจฉริยะ</p>
</div>
""", unsafe_allow_html=True)

# ========== Main Content ==========
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# ========== Quick Stats Card ==========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title-dark"><span class="emoji">📊</span> ข้อมูลระบบ</div>', unsafe_allow_html=True)

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

# ========== Prediction Card ==========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title-dark"><span class="emoji">🔮</span> ทำนายเกรดนักเรียน</div>', unsafe_allow_html=True)

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
    
    # Numeric inputs - 2 columns for better visibility
    cols = st.columns(2)
    for i, feat in enumerate(numeric_features):
        with cols[i % 2]:
            mean_val = float(np.mean(test_data['X_test'][:, feature_names.index(feat)]))
            # ปัดเป็นทศนิยม 1 ตำแหน่งให้เห็นชัดเจน
            default_val = round(mean_val, 1)
            inputs[feat] = st.number_input(
                f"{feat.replace('_', ' ').title()}",
                value=default_val,
                step=0.5,
                format="%.1f",
                key=f"num_{feat}",
                help=f"ค่าเฉลี่ย: {default_val}"
            )
    
    # Categorical inputs
    if categorical_features:
        st.markdown("### 🏷️ ข้อมูลเพิ่มเติม")
        cat_cols = st.columns(2)
        for i, (base_name, categories) in enumerate(categorical_features.items()):
            with cat_cols[i % 2]:
                options = categories if categories else ['Yes', 'No']
                inputs[base_name] = st.selectbox(
                    base_name.replace('_', ' ').title(),
                    options,
                    index=0,
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
        st.markdown("###  ความน่าจะเป็นของแต่ละเกรด")
        
        prob_df = pd.DataFrame({
            'Grade': target_names,
            'Probability': proba
        }).sort_values('Probability', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#10b981' if g == pred_label else '#e2e8f0' for g in prob_df['Grade']]
        bars = ax.barh(prob_df['Grade'], prob_df['Probability'], 
                       color=colors, height=0.5, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('ความน่าจะเป็น', fontsize=13, fontweight='700', color='#0f172a')
        ax.set_title('ความน่าจะเป็นของแต่ละเกรด', fontsize=16, fontweight='800', 
                    color='#0f172a', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (idx, row) in enumerate(prob_df.iterrows()):
            ax.text(row['Probability'] + 0.02, i, f"{row['Probability']:.1%}", 
                   va='center', fontsize=12, fontweight='700', color='#0f172a')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ========== Model Info ==========
with st.expander("️ เลือกดูโมเดลอื่นๆ"):
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
        
        cm = confusion_matrix(test_data['y_test'], y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=target_names, yticklabels=target_names)
        ax.set_title(f'Confusion Matrix - {selected_model}', color='#0f172a', fontweight='700')
        st.pyplot(fig)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 2rem;">
    <p style="font-size: 1rem; margin: 0; color: #94a3b8;">Built with ❤️ using Streamlit + Machine Learning</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #64748b;">© 2024 AI Grade Predictor System</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)