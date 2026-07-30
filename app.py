import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
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
    .prediction-result {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white; padding: 1.5rem; border-radius: 16px;
        text-align: center; font-size: 1.3rem; font-weight: 600; margin-top: 1rem;
    }
    .upload-box {
        background: white; padding: 2rem; border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ========== Header ==========
st.markdown('<h1 class="main-title"> Student Grade Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ระบบทำนายเกรดนักเรียนด้วย Machine Learning</p>', unsafe_allow_html=True)

# ========== Upload Models ==========
st.markdown("### 📦 อัปโหลดไฟล์โมเดล")
st.markdown("""
<div class="upload-box">
    <p>กรุณาอัปโหลดไฟล์ <code>.pkl</code> ทั้งหมดจากโฟลเดอร์ <code>models/</code></p>
    <p style="color: #6b7280; font-size: 0.9rem;">
        (K-Means.pkl, SVM.pkl, Decision_Tree.pkl, KNN.pkl, Logistic_Regression.pkl, 
        Random_Forest.pkl, scaler.pkl, test_data.pkl)
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "เลือกไฟล์ทั้งหมด (可以多选)",
    type=['pkl'],
    accept_multiple_files=True
)

if not uploaded_files:
    st.warning("⚠️ ยังไม่มีการอัปโหลดไฟล์โมเดล กรุณาอัปโหลดไฟล์จาก Colab ก่อน")
    st.stop()

# ========== Load Models from Uploaded Files ==========
@st.cache_resource
def load_from_uploaded(files):
    models = {}
    scaler = None
    test_data = None
    
    for f in files:
        # ใช้ชื่อไฟล์เป็นตัวระบุ
        fname = f.name
        f.seek(0)
        data = joblib.load(f)
        
        if fname == 'scaler.pkl':
            scaler = data
        elif fname == 'test_data.pkl':
            test_data = data
        else:
            name = fname.replace('_', ' ').replace('.pkl', '')
            models[name] = data
    
    return models, scaler, test_data

models, scaler, test_data = load_from_uploaded(uploaded_files)

# ตรวจสอบว่ามีไฟล์ครบไหม
required_files = ['scaler.pkl', 'test_data.pkl']
uploaded_names = [f.name for f in uploaded_files]
missing = [f for f in required_files if f not in uploaded_names]

if missing:
    st.error(f" ขาดไฟล์: {missing}")
    st.stop()

if not models:
    st.error("❌ ไม่พบไฟล์โมเดล (K-Means.pkl, SVM.pkl, ฯลฯ)")
    st.stop()

st.success(f"✅ โหลดโมเดลสำเร็จ! ({len(models)} โมเดล)")

# ========== Sidebar ==========
st.sidebar.markdown("## 🎛️ Control Panel")
model_name = st.sidebar.selectbox("🧠 เลือกโมเดล", list(models.keys()))
model = models[model_name]

feature_names = test_data['feature_names']
target_names = test_data['target_names']

# ========== Main Content ==========
st.markdown(f'<div class="section-title">📊 ผลการทดสอบ: {model_name}</div>', unsafe_allow_html=True)

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
    st.markdown("#### 📈 Confusion Matrix")
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
st.markdown('<div class="section-title">🔮 ทดลองทำนายเกรดนักเรียนใหม่</div>', unsafe_allow_html=True)

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
    st.markdown("##### 📋 กรอกข้อมูลนักเรียน")
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
st.markdown('<div style="text-align:center; color:#9ca3af; padding:1rem;">Built with ❤️ using Streamlit + Scikit-learn</div>', unsafe_allow_html=True)