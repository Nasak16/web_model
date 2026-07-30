import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
# 🎨 Page Configuration
# ============================================

st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Simple & Beautiful
st.markdown("""
<style>
    .main {
        background-color: #fafafa;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
#  Load Models
# ============================================

@st.cache_resource
def load_all_models():
    """โหลดโมเดลทั้งหมดจากโฟลเดอร์ models/"""
    models = {}
    model_dir = 'models'
    
    if not os.path.exists(model_dir):
        return None
    
    files_to_load = {
        'knn': 'knn_model.pkl',
        'dt': 'decision_tree_model.pkl',
        'svm': 'svm_model.pkl',
        'rf': 'random_forest_model.pkl',
        'lr': 'linear_regression_model.pkl',
        'kmeans': 'kmeans_model.pkl',
        'scaler': 'scaler.pkl',
        'info': 'model_info.pkl',
        'le': 'label_encoder.pkl'
    }
    
    for key, filename in files_to_load.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)
    
    return models if len(models) > 0 else None

models = load_all_models()

# ============================================
#  Sidebar
# ============================================

st.sidebar.title("🤖 ML Prediction App")
st.sidebar.markdown("---")

if models:
    app_mode = st.sidebar.radio(
        "เลือกโมเดล",
        ["📊 Dashboard", "🎯 KNN", " Decision Tree", 
         "⚡ SVM", "🌲 Random Forest", "📈 Regression", "🔵 K-Means"]
    )
else:
    st.sidebar.error("⚠️ ไม่พบไฟล์โมเดล")
    st.sidebar.info("📁 วางไฟล์ .pkl ทั้งหมดในโฟลเดอร์ `models/`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 อัพโหลดไฟล์ Excel")
uploaded_file = st.sidebar.file_uploader(
    "เลือกไฟล์ .xlsx หรือ .csv",
    type=['xlsx', 'xls', 'csv']
)

# ============================================
# 📊 Dashboard
# ============================================

if app_mode == "📊 Dashboard":
    st.title("📊 Machine Learning Dashboard")
    st.markdown("### ภาพรวมประสิทธิภาพของโมเดล")
    
    if models and 'info' in models:
        info = models['info']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("KNN", f"{info['scores']['KNN']:.2%}")
        with col2:
            st.metric("SVM", f"{info['scores']['SVM']:.2%}")
        with col3:
            st.metric("Decision Tree", f"{info['scores']['Decision Tree']:.2%}")
        with col4:
            st.metric("Random Forest", f"{info['scores']['Random Forest']:.2%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            scores = info['scores']
            ax.barh(list(scores.keys()), list(scores.values()), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
            ax.set_xlim([0, 1])
            ax.set_xlabel('Score')
            ax.set_title('Model Performance')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("###  ข้อมูลโมเดล")
            st.write(f"**Task:** {info['task']}")
            st.write(f"**Target:** {info['target_column']}")
            st.write(f"**Features:** {len(info['feature_names'])} ตัว")
            if info['classes']:
                st.write(f"**Classes:** {info['classes']}")
    else:
        st.warning("⚠️ ไม่พบไฟล์ model_info.pkl")

# ============================================
# 🎯 KNN
# ============================================

elif app_mode == "🎯 KNN":
    st.title("🎯 K-Nearest Neighbor")
    
    if models and all(k in models for k in ['knn', 'scaler', 'info']):
        info = models['info']
        features = info['feature_names']
        
        st.markdown("### กรอกข้อมูลเพื่อทำนาย")
        
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                input_data[feat] = st.number_input(feat, value=0.0, key=f'knn_{feat}')
        
        if st.button("🔮 ทำนายผล"):
            df_input = pd.DataFrame([input_data])
            # จัดคอลัมน์ให้ตรงกับตอน train
            df_input = df_input.reindex(columns=features, fill_value=0)
            scaled = models['scaler'].transform(df_input)
            pred = models['knn'].predict(scaled)
            
            if info['classes']:
                result = info['classes'][pred[0]]
                st.success(f"✅ ผลการทำนาย: **{result}**")
            else:
                st.success(f"✅ ผลการทำนาย: **{pred[0]:.4f}**")
    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล KNN")

# ============================================
# 🌳 Decision Tree
# ============================================

elif app_mode == "🌳 Decision Tree":
    st.title("🌳 Decision Tree")
    
    if models and all(k in models for k in ['dt', 'scaler', 'info']):
        info = models['info']
        features = info['feature_names']
        
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                input_data[feat] = st.number_input(feat, value=0.0, key=f'dt_{feat}')
        
        if st.button("🔮 ทำนายผล"):
            df_input = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
            scaled = models['scaler'].transform(df_input)
            pred = models['dt'].predict(scaled)
            
            if info['classes']:
                st.success(f"✅ ผลการทำนาย: **{info['classes'][pred[0]]}**")
            else:
                st.success(f"✅ ผลการทำนาย: **{pred[0]:.4f}**")
            
            # Feature Importance
            if hasattr(models['dt'], 'feature_importances_'):
                st.markdown("### 📊 ความสำคัญของ Features")
                imp_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': models['dt'].feature_importances_
                }).sort_values('Importance', ascending=True)
                st.bar_chart(imp_df.set_index('Feature'))
    else:
        st.error("️ ไม่พบไฟล์โมเดล")

# ============================================
# ⚡ SVM
# ============================================

elif app_mode == "⚡ SVM":
    st.title("⚡ Support Vector Machine")
    
    if models and all(k in models for k in ['svm', 'scaler', 'info']):
        info = models['info']
        features = info['feature_names']
        
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                input_data[feat] = st.number_input(feat, value=0.0, key=f'svm_{feat}')
        
        if st.button("🔮 ทำนายผล"):
            df_input = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
            scaled = models['scaler'].transform(df_input)
            pred = models['svm'].predict(scaled)
            
            if info['classes']:
                st.success(f"✅ ผลการทำนาย: **{info['classes'][pred[0]]}**")
            else:
                st.success(f"✅ ผลการทำนาย: **{pred[0]:.4f}**")
    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล")

# ============================================
# 🌲 Random Forest
# ============================================

elif app_mode == " Random Forest":
    st.title(" Random Forest (Ensemble)")
    
    if models and all(k in models for k in ['rf', 'scaler', 'info']):
        info = models['info']
        features = info['feature_names']
        
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                input_data[feat] = st.number_input(feat, value=0.0, key=f'rf_{feat}')
        
        if st.button("🔮 ทำนายผล"):
            df_input = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
            scaled = models['scaler'].transform(df_input)
            pred = models['rf'].predict(scaled)
            
            if info['classes']:
                st.success(f"✅ ผลการทำนาย: **{info['classes'][pred[0]]}**")
            else:
                st.success(f"✅ ผลการทำนาย: **{pred[0]:.4f}**")
            
            if hasattr(models['rf'], 'feature_importances_'):
                st.markdown("### 📊 Feature Importance")
                imp_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': models['rf'].feature_importances_
                }).sort_values('Importance', ascending=True)
                st.bar_chart(imp_df.set_index('Feature'))
    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล")

# ============================================
# 📈 Regression
# ============================================

elif app_mode == "📈 Regression":
    st.title("📈 Linear Regression")
    
    if models and all(k in models for k in ['lr', 'scaler', 'info']):
        info = models['info']
        features = info['feature_names']
        
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                input_data[feat] = st.number_input(feat, value=0.0, key=f'lr_{feat}')
        
        if st.button("🔮 ทำนายค่า"):
            df_input = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
            scaled = models['scaler'].transform(df_input)
            pred = models['lr'].predict(scaled)
            st.success(f"✅ ค่าที่ทำนายได้: **{pred[0]:.4f}**")
    else:
        st.error("️ ไม่พบไฟล์โมเดล")

# ============================================
# 🔵 K-Means
# ============================================

elif app_mode == "🔵 K-Means":
    st.title("🔵 K-Means Clustering")
    
    if models and all(k in models for k in ['kmeans', 'scaler', 'info']):
        info = models['info']
        features = info['feature_names']
        
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                input_data[feat] = st.number_input(feat, value=0.0, key=f'km_{feat}')
        
        if st.button("🔮 จัดกลุ่ม"):
            df_input = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
            scaled = models['scaler'].transform(df_input)
            cluster = models['kmeans'].predict(scaled)
            st.success(f"✅ ข้อมูลนี้อยู่ใน Cluster: **{cluster[0]}**")
        
        # Visualize clusters
        if uploaded_file:
            st.markdown("---")
            st.markdown("### 📊 Visualization จากไฟล์ที่อัพโหลด")
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            X = df.drop(columns=[info['target_column']])
            X = pd.get_dummies(X, drop_first=True)
            X_scaled = models['scaler'].transform(X)
            labels = models['kmeans'].predict(X_scaled)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                c=labels, cmap='viridis', alpha=0.6)
            ax.scatter(models['kmeans'].cluster_centers_[:, 0], 
                      models['kmeans'].cluster_centers_[:, 1],
                      c='red', s=200, marker='X', label='Centroids')
            ax.set_xlabel(features[0] if len(features) > 0 else 'Feature 1')
            ax.set_ylabel(features[1] if len(features) > 1 else 'Feature 2')
            ax.legend()
            ax.set_title('K-Means Clustering')
            st.pyplot(fig)
    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล")

# ============================================
# 📊 แสดงข้อมูลจาก Excel
# ============================================

if uploaded_file is not None:
    st.sidebar.success(f"✅ โหลดไฟล์: {uploaded_file.name}")
    
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.markdown("---")
    st.markdown("### 📄 ข้อมูลจากไฟล์ Excel")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 ขนาดข้อมูล")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
    with col2:
        st.markdown("### 📈 สรุปสถิติ")
        st.dataframe(df.describe())

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #95a5a6;'><p>Created with ❤️ using Streamlit</p></div>", unsafe_allow_html=True)