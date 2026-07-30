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
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== Custom CSS - New Design ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Main container */
    .main-container {
        background: white;
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1rem auto;
        max-width: 900px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #667eea;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .prediction-box h2 {
        color: white;
        font-size: 1.3rem;
        margin: 0 0 1rem 0;
        font-weight: 600;
    }
    
    .prediction-result {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .prediction-label {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* Input fields styling */
    .stNumberInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 12px;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metrics cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Form container */
    .form-container {
        background: #f8f9ff;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: white;
        padding: 2rem;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .main-container { padding: 1.5rem; margin: 0.5rem; }
        .prediction-result { font-size: 2.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ========== Load Models ==========
@st.cache_resource
def load_all():
    models_path = 'models'
    
    if not os.path.exists(models_path):
        st.error("❌ ไม่พบโฟลเดอร์ 'models'")
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
    st.stop()

feature_names = test_data['feature_names']
target_names = test_data['target_names']

# ========== Header ==========
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Grade Predictor</h1>
    <p>ระบบทำนายเกรดนักเรียนด้วย Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ========== Main Container ==========
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Section Title
st.markdown('<div class="section-title">🔮 ทำนายเกรดนักเรียน</div>', unsafe_allow_html=True)

# แยก features
numeric_features = []
categorical_features = {}

for feat in feature_names:
    if '_' in feat:
        base, val = feat.rsplit('_', 1)
        if base not in categorical_features:
            categorical_features[base] = []
        categorical_features[base].append(val)
    else:
        numeric_features.append(feat)

# Create form
with st.form("prediction_form", clear_on_submit=False):
    st.markdown("### 📋 กรอกข้อมูลนักเรียน")
    st.markdown("กรอกข้อมูลด้านล่างเพื่อทำนายเกรด")
    
    inputs = {}
    
    # Numeric inputs - 2 columns
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    num_cols = st.columns(2)
    
    for i, feat in enumerate(numeric_features):
        with num_cols[i % 2]:
            mean_val = float(np.mean(test_data['X_test'][:, feature_names.index(feat)]))
            inputs[feat] = st.number_input(
                f"{feat.replace('_', ' ').title()}",
                value=round(mean_val, 2),
                step=0.1,
                format="%.2f",
                key=f"num_{feat}"
            )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Categorical inputs
    if categorical_features:
        st.markdown("#### หมวดหมู่")
        cat_cols = st.columns(2)
        for i, (base_name, categories) in enumerate(categorical_features.items()):
            with cat_cols[i % 2]:
                options = ['None'] + categories
                inputs[base_name] = st.selectbox(
                    base_name.replace('_', ' ').title(),
                    options,
                    key=f"cat_{base_name}"
                )
    
    # Submit button
    submitted = st.form_submit_button("🚀 ทำนายเกรดตอนนี้", use_container_width=True)

if submitted:
    # Prepare input
    input_df = pd.DataFrame([inputs])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)
    
    # Use best model (Random Forest if available)
    model_name = 'Random Forest' if 'Random Forest' in models else list(models.keys())[0]
    model = models[model_name]
    
    # Predict
    pred = model.predict(input_scaled)[0]
    pred_label = target_names[int(pred)] if int(pred) < len(target_names) else str(pred)
    
    # Show result with animation
    st.markdown(f"""
    <div class="prediction-box">
        <h2>🎯 เกรดที่ทำนายได้</h2>
        <div class="prediction-result">{pred_label}</div>
        <div class="prediction-label">ใช้โมเดล: {model_name}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show probabilities if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        
        st.markdown("### 📊 ความน่าจะเป็นของแต่ละเกรด")
        
        # Create horizontal bar chart
        prob_df = pd.DataFrame({
            'Grade': target_names,
            'Probability': proba
        }).sort_values('Probability', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#10b981' if g == pred_label else '#e0e0e0' for g in prob_df['Grade']]
        bars = ax.barh(prob_df['Grade'], prob_df['Probability'], color=colors, height=0.6)
        
        ax.set_xlabel('ความน่าจะเป็น', fontsize=12, fontweight='bold')
        ax.set_title('ความน่าจะเป็นของแต่ละเกรด', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, (idx, row) in enumerate(prob_df.iterrows()):
            ax.text(row['Probability'] + 0.02, i, f"{row['Probability']:.1%}", 
                   va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ========== Model Info (Optional) ==========
with st.expander("📚 ข้อมูลเกี่ยวกับโมเดล"):
    st.markdown(f"""
    ### โมเดลที่พร้อมใช้งาน ({len(models)} โมเดล)
    
    {', '.join(sorted(models.keys()))}
    
    **จำนวน features:** {len(feature_names)}
    
    **จำนวนคลาส:** {len(target_names)} ({', '.join(target_names)})
    """)

# ========== Footer ==========
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using Streamlit + Machine Learning</p>
    <p>© 2024 Student Grade Predictor</p>
</div>
""", unsafe_allow_html=True)