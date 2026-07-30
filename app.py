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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%); }
    .main-title {
        text-align: center; font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .subtitle { text-align: center; color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem; }
    .section-title {
        font-size: 1.4rem; font-weight: 600; color: #1e293b;
        margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .metric-box {
        background: white; padding: 1.2rem; border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
    }
    .prediction-result {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white; padding: 1.5rem; border-radius: 16px;
        text-align: center; font-size: 1.3rem; font-weight: 600; margin-top: 1rem;
    }
    .footer { text-align: center; color: #9ca3af; padding: 2rem 0 1rem 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ========== Load Models from 'models/' folder ==========
@st.cache_resource
def load_all():
    models_path = 'models'
    
    if not os.path.exists(models_path):
        st.error(f"❌ ไม่พบโฟลเดอร์ '{models_path}' กรุณาตรวจสอบว่ามีการอัปโหลดโฟลเดอร์ models/ ขึ้น GitHub แล้ว")
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

# ตรวจสอบว่าโหลดสำเร็จไหม
if models is None:
    st.stop()

if not models:
    st.error("❌ ไม่พบไฟล์โมเดลในโฟลเดอร์ models/ กรุณาตรวจสอบว่ามีไฟล์ .pkl ครบถ้วน")
    st.stop()

if test_data is None:
    st.error("❌ ไม่พบไฟล์ test_data.pkl")
    st.stop()

feature_names = test_data['feature_names']
target_names = test_data['target_names']

# ========== Header ==========
st.markdown('<h1 class="main-title">🎓 Student Grade Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ระบบทำนายเกรดนักเรียนด้วย Machine Learning</p>', unsafe_allow_html=True)

# ========== Sidebar ==========
st.sidebar.markdown("## 🎛️ Control Panel")
model_name = st.sidebar.selectbox("🧠 เลือกโมเดล", sorted(list(models.keys())))
model = models[model_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 โมเดลที่พร้อมใช้งาน")
for m in sorted(models.keys()):
    st.sidebar.markdown(f"• {m}")

# ========== Main Content ==========
st.markdown(f'<div class="section-title"> ผลการทดสอบ: {model_name}</div>', unsafe_allow_html=True)

# --- Metrics ---
if model_name == 'K-Means':
    from sklearn.metrics import silhouette_score
    labels = model.predict(test_data['X_test'])
    score = silhouette_score(test_data['X_test'], labels)
    col1, col2, col3 = st.columns(3)
    col1.metric("🔷 Silhouette Score", f"{score:.4f}")
    col2.metric("จำนวน Clusters", model.n_clusters)
    col3.metric("📝 จำนวนข้อมูล Test", len(labels))
else:
    y_pred = model.predict(test_data['X_test'])
    acc = accuracy_score(test_data['y_test'], y_pred)
    correct = sum(y_pred == test_data['y_test'])
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy", f"{acc*100:.2f}%")
    col2.metric("✅ ทำนายถูก", f"{correct}/{len(y_pred)}")
    col3.metric("📝 จำนวนข้อมูล Test", len(y_pred))
    
    # Confusion Matrix
    st.markdown("####  Confusion Matrix")
    cm = confusion_matrix(test_data['y_test'], y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                linewidths=2, linecolor='white')
    ax.set_xlabel('Predicted Grade', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Grade', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig)

# Feature Importance
if model_name in ['Random Forest', 'Decision Tree'] and hasattr(model, 'feature_importances_'):
    st.markdown("#### 🌟 Feature Importance")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(importances)), importances[sorted_idx], color='#667eea', edgecolor='white')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ========== Live Prediction ==========
st.markdown('<div class="section-title"> ทดลองทำนายเกรดนักเรียนใหม่</div>', unsafe_allow_html=True)

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

with st.form("prediction_form"):
    st.markdown("#####  กรอกข้อมูลนักเรียน")
    inputs = {}
    
    num_cols = st.columns(min(4, len(numeric_features)))
    for i, feat in enumerate(numeric_features):
        with num_cols[i % len(num_cols)]:
            mean_val = float(np.mean(test_data['X_test'][:, feature_names.index(feat)]))
            inputs[feat] = st.number_input(
                feat.replace('_', ' ').title(),
                value=round(mean_val, 2), step=0.1, format="%.2f"
            )
    
    if categorical_features:
        st.markdown("###### ตัวเลือก (Categorical)")
        cat_cols = st.columns(min(3, len(categorical_features)))
        for i, (base_name, categories) in enumerate(categorical_features.items()):
            with cat_cols[i % len(cat_cols)]:
                options = ['None'] + categories
                inputs[base_name] = st.selectbox(
                    base_name.replace('_', ' ').title(), options
                )
    
    submitted = st.form_submit_button("🚀 ทำนายเกรด", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([inputs])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    pred_label = target_names[int(pred)] if int(pred) < len(target_names) else str(pred)
    
    st.markdown(f"""
    <div class="prediction-result">
        🎯 เกรดที่ทำนายได้: <strong>{pred_label}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        st.markdown("##### 📊 ความน่าจะเป็นของแต่ละเกรด")
        prob_df = pd.DataFrame({
            'Grade': target_names,
            'Probability': proba
        }).sort_values('Probability', ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(prob_df['Grade'][::-1], prob_df['Probability'][::-1], 
                color='#667eea', edgecolor='white')
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown('<div class="footer">Built with ❤️ using Streamlit + Scikit-learn | Student Grade Prediction System</div>', unsafe_allow_html=True)