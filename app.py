import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        padding: 12px 30px; font-weight: 600; font-size: 16px;
    }
    .developer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 15px; color: white; text-align: center;
    }
    .model-card {
        background: #1e1e2e; padding: 20px; border-radius: 12px;
        border-left: 4px solid #667eea; margin: 10px 0;
    }
    .grade-A { background: linear-gradient(135deg, #4CAF50, #8BC34A) !important; }
    .grade-B { background: linear-gradient(135deg, #2196F3, #03A9F4) !important; }
    .grade-C { background: linear-gradient(135deg, #FFC107, #FF9800) !important; }
    .grade-D { background: linear-gradient(135deg, #FF5722, #F44336) !important; }
    .grade-F { background: linear-gradient(135deg, #9E9E9E, #607D8B) !important; }
    h1, h2, h3 { color: #ffffff; }
    .param-box {
        background: #1e1e2e; padding: 15px; border-radius: 10px; margin: 10px 0;
        border: 1px solid #333; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# ส่วนข้อมูลผู้พัฒนา
# ============================================
def show_developer_info():
    st.markdown("### 👨‍💻 ข้อมูลผู้พัฒนา")
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_image = st.file_uploader("อัพโหลดรูปผู้พัฒนา", type=['png', 'jpg', 'jpeg'])
        if uploaded_image:
            st.image(Image.open(uploaded_image), width=200)
        else:
            st.image("https://via.placeholder.com/200x200.png?text=Profile+Picture", width=200)
    with col2:
        name = st.text_input("ชื่อ-นามสกุล", "นาย สมชาย ใจดี")
        student_id = st.text_input("รหัสนักศึกษา", "6501234567")
        group = st.text_input("หมู่เรียน", "CS-2A")
        st.markdown(f"""
        <div class="developer-card">
            <h2 style="color: white; margin: 0;">{name}</h2>
            <p style="color: white; margin: 10px 0; font-size: 18px;">รหัส: {student_id}</p>
            <p style="color: white; margin: 10px 0; font-size: 18px;">หมู่เรียน: {group}</p>
        </div>
        """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_dir = 'models'
    if os.path.exists(model_dir):
        files = {
            'knn': 'knn_model.pkl', 'dt': 'decision_tree_model.pkl',
            'svm': 'svm_model.pkl', 'rf': 'random_forest_model.pkl',
            'lr': 'linear_regression_model.pkl', 'kmeans': 'kmeans_model.pkl',
            'scaler': 'scaler.pkl', 'info': 'model_info.pkl',
            'X_train': 'X_train_scaled.pkl', 'y_train': 'y_train.pkl'
        }
        for key, filename in files.items():
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                models[key] = joblib.load(path)
    return models

models = load_models()
st.title("🎓 Student Grade Prediction System")

# Navigation
page = st.sidebar.selectbox(
    "📌 เลือกหน้า",
    ["👨‍💻 ข้อมูลผู้พัฒนา", "🔮 ทำนายผล", "📊 เปรียบเทียบโมเดล", "🔵 K-Means Clustering", "📈 ข้อมูลโมเดล"]
)

# ============================================
# หน้า 1: ข้อมูลผู้พัฒนา
# ============================================
if page == "👨‍💻 ข้อมูลผู้พัฒนา":
    show_developer_info()
    st.markdown("---")
    st.markdown("### 📋 เกี่ยวกับโปรเจค")
    st.markdown("""
    **Student Grade Prediction System**
    โปรเจคนี้พัฒนาเพื่อทำนายเกรดนักเรียนจากปัจจัยต่างๆ โดยใช้ Machine Learning 6 โมเดล
    
    **เทคโนโลยีที่ใช้:** Python, Scikit-Learn, Streamlit, Pandas, Matplotlib
    """)

# ============================================
# หน้า 2: ทำนายผล (ปรับปรุง Layout)
# ============================================
elif page == "🔮 ทำนายผล":
    st.markdown("### 🎯 ทำนายเกรดด้วยโมเดลต่างๆ")
    
    # เลือกโมเดล
    model_choice = st.selectbox("เลือกโมเดล", ["K-Nearest Neighbor", "Decision Tree", "SVM", "Random Forest", "Linear Regression"])
    
    # สำหรับ KNN แสดง Slider เลือกค่า K ตั้งแต่ต้น
    if model_choice == "K-Nearest Neighbor":
        st.markdown("### ⚙️ ตั้งค่าพารามิเตอร์ KNN")
        k_value = st.slider(
            "🔢 เลือกจำนวน K (Neighbors)",
            min_value=1,
            max_value=20,
            value=getattr(models['knn'], 'n_neighbors', 5) if 'knn' in models else 5,
            step=1
        )
        st.markdown("---")
    
    # ฟอร์มกรอกข้อมูล - แบ่งเป็น 2 คอลัมน์
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.number_input(" Student ID", min_value=0, value=1001, step=1)
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        study_time_hours = st.number_input("⏰ Study Time (hours/day)", min_value=0.0, max_value=12.0, value=6.5, step=0.5)
        attendance_percent = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
        sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        parental_education = st.selectbox("🎓 Parental Education", ["None", "High School", "Bachelors", "Masters", "PhD"])
    
    with col2:
        internet_access = st.selectbox("🌐 Internet Access", ["Yes", "No"])
        extracurricular_activities = st.selectbox("🎯 Extracurricular", ["Yes", "No"])
        part_time_job = st.selectbox("💼 Part-time Job", ["Yes", "No"])
        previous_grade = st.number_input("📚 Previous Grade", min_value=0.0, max_value=100.0, value=88.5, step=0.1)
        final_exam_score = st.number_input("📝 Final Exam Score", min_value=0.0, max_value=100.0, value=92.0, step=0.1)
    
    if st.button(" ทำนายผล", width='stretch'):
        input_data = pd.DataFrame({
            'student_id': [student_id], 'gender': [gender], 'study_time_hours': [study_time_hours],
            'attendance_percent': [attendance_percent], 'sleep_hours': [sleep_hours],
            'parental_education': [parental_education], 'internet_access': [internet_access],
            'extracurricular_activities': [extracurricular_activities], 'part_time_job': [part_time_job],
            'previous_grade': [previous_grade], 'final_exam_score': [final_exam_score]
        })
        
        st.markdown("### 📋 ข้อมูลที่กรอก")
        st.dataframe(input_data, width='stretch')
        
        if models and 'scaler' in models:
            try:
                feature_names = models['info'].get('feature_names', [])
                input_processed = pd.get_dummies(input_data, drop_first=True)
                for col in feature_names:
                    if col not in input_processed.columns:
                        input_processed[col] = 0
                input_processed = input_processed[feature_names]
                scaled_data = models['scaler'].transform(input_processed)
                
                model_map = {"K-Nearest Neighbor": 'knn', "Decision Tree": 'dt', "SVM": 'svm', "Random Forest": 'rf', "Linear Regression": 'lr'}
                model_key = model_map[model_choice]
                
                if model_key in models:
                    start_time = time.time()
                    
                    # สำหรับ KNN ใช้ค่า K ที่เลือก
                    if model_choice == "K-Nearest Neighbor" and 'X_train' in models and 'y_train' in models:
                        with st.spinner(f'🔄 กำลัง retrain โมเดลด้วย K={k_value}...'):
                            from sklearn.neighbors import KNeighborsClassifier
                            active_model = KNeighborsClassifier(
                                n_neighbors=k_value, 
                                metric=models['knn'].metric, 
                                weights=models['knn'].weights
                            )
                            active_model.fit(models['X_train'], models['y_train'])
                            prediction = active_model.predict(scaled_data)
                            exec_time = time.time() - start_time
                            st.success(f"✅ Retrain สำเร็จด้วย K={k_value}")
                    else:
                        prediction = models[model_key].predict(scaled_data)
                        exec_time = time.time() - start_time
                        active_model = models[model_key]
                    
                    st.markdown("---")
                    st.markdown(f"### 🎯 ผลการทำนายจาก {model_choice}")
                    
                    classes = models['info'].get('classes', ['A', 'B', 'C', 'D', 'F'])
                    
                    # --- 1. KNN Display ---
                    if model_choice == "K-Nearest Neighbor":
                        grade = classes[int(prediction[0])] if int(prediction[0]) < len(classes) else str(prediction[0])
                        
                        st.markdown(f"""
                        <div class="metric-card grade-{grade}" style="padding: 40px; text-align: center; border-radius: 15px;">
                            <h1 style="color: white; margin: 0; font-size: 96px; font-weight: bold;">{grade}</h1>
                            <p style="color: white; margin: 10px 0; font-size: 18px;">เกรดที่ทำนาย (K-Nearest Neighbor, K={k_value})</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("💡 **KNN** ทำนายโดยดูจากเพื่อนบ้านที่ใกล้ที่สุด (K neighbors)")
                        
                        if hasattr(active_model, 'predict_proba'):
                            proba = active_model.predict_proba(scaled_data)[0]
                            st.markdown("### 📊 ความน่าจะเป็นของแต่ละเกรด")
                            for cls, prob in zip(classes, proba):
                                pct = int(prob * 100)
                                bar_len = int(prob * 30)
                                bar = "█" * bar_len + "░" * (30 - bar_len)
                                color = "#4CAF50" if cls == grade else "#95a5a6"
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; margin: 8px 0; padding: 8px; background: #1e1e2e; border-radius: 8px;">
                                    <span style="color: white; font-weight: bold; width: 30px;">{cls}:</span>
                                    <span style="color: {color}; margin: 0 10px; font-family: monospace; font-size: 14px;">{bar}</span>
                                    <span style="color: white; font-weight: bold;">{pct}%</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # แสดงพารามิเตอร์
                        st.markdown("### 📋 พารามิเตอร์ปัจจุบัน")
                        p_col1, p_col2, p_col3 = st.columns(3)
                        with p_col1:
                            st.markdown(f'<div class="param-box"><span style="color:#95a5a6">🔢 K (Neighbors):</span><br><span style="color:#667eea; font-size:24px; font-weight:bold">{k_value}</span></div>', unsafe_allow_html=True)
                        with p_col2:
                            metric = getattr(active_model, 'metric', 'N/A')
                            st.markdown(f'<div class="param-box"><span style="color:#95a5a6">📐 Metric:</span><br><span style="color:#667eea; font-size:20px; font-weight:bold">{metric}</span></div>', unsafe_allow_html=True)
                        with p_col3:
                            weights = getattr(active_model, 'weights', 'N/A')
                            st.markdown(f'<div class="param-box"><span style="color:#95a5a6">⚖️ Weights:</span><br><span style="color:#667eea; font-size:20px; font-weight:bold">{weights}</span></div>', unsafe_allow_html=True)

                    # --- 2. Decision Tree Display ---
                    elif model_choice == "Decision Tree":
                        grade = classes[int(prediction[0])] if int(prediction[0]) < len(classes) else str(prediction[0])
                        st.markdown(f"""
                        <div class="metric-card grade-{grade}" style="padding: 40px; text-align: center; border-radius: 15px;">
                            <h1 style="color: white; margin: 0; font-size: 96px; font-weight: bold;">{grade}</h1>
                            <p style="color: white; margin: 10px 0; font-size: 18px;">เกรดที่ทำนาย (Decision Tree)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("🌳 **Decision Tree** ตัดสินใจตามเงื่อนไขของ features แบบต้นไม้")
                        if hasattr(models[model_key], 'feature_importances_'):
                            st.markdown("### 📊 ความสำคัญของ Features (Feature Importance)")
                            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': models[model_key].feature_importances_}).sort_values('Importance', ascending=True)
                            st.bar_chart(imp_df.set_index('Feature'))

                    # --- 3. SVM Display ---
                    elif model_choice == "SVM":
                        grade = classes[int(prediction[0])] if int(prediction[0]) < len(classes) else str(prediction[0])
                        st.markdown(f"""
                        <div class="metric-card grade-{grade}" style="padding: 40px; text-align: center; border-radius: 15px;">
                            <h1 style="color: white; margin: 0; font-size: 96px; font-weight: bold;">{grade}</h1>
                            <p style="color: white; margin: 10px 0; font-size: 18px;">เกรดที่ทำนาย (SVM)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("⚡ **SVM** หาเส้นแบ่ง (Hyperplane) ที่แยก classes ได้ดีที่สุด")
                        if hasattr(models[model_key], 'decision_function'):
                            conf = models[model_key].decision_function(scaled_data)[0]
                            st.markdown(f'<div class="param-box"><span style="color:#95a5a6">🎯 Confidence Score:</span><br><span style="color:#667eea; font-size:24px; font-weight:bold">{conf[0]:.4f}</span></div>', unsafe_allow_html=True)

                    # --- 4. Random Forest Display ---
                    elif model_choice == "Random Forest":
                        grade = classes[int(prediction[0])] if int(prediction[0]) < len(classes) else str(prediction[0])
                        st.markdown(f"""
                        <div class="metric-card grade-{grade}" style="padding: 40px; text-align: center; border-radius: 15px;">
                            <h1 style="color: white; margin: 0; font-size: 96px; font-weight: bold;">{grade}</h1>
                            <p style="color: white; margin: 10px 0; font-size: 18px;">เกรดที่ทำนาย (Random Forest)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("🌲 **Random Forest** โหวตผลลัพธ์จาก Decision Trees หลายร้อยต้น")
                        if hasattr(models[model_key], 'n_estimators'):
                            st.markdown(f'<div class="param-box"><span style="color:#95a5a6">🌳 จำนวน Trees:</span><br><span style="color:#667eea; font-size:24px; font-weight:bold">{models[model_key].n_estimators}</span></div>', unsafe_allow_html=True)
                        if hasattr(models[model_key], 'feature_importances_'):
                            st.markdown("### 📊 ความสำคัญของ Features")
                            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': models[model_key].feature_importances_}).sort_values('Importance', ascending=True)
                            st.bar_chart(imp_df.set_index('Feature'))

                    # --- 5. Linear Regression Display ---
                    elif model_choice == "Linear Regression":
                        raw_score = prediction[0]
                        
                        if raw_score < 5: 
                            nearest_idx = int(round(raw_score))
                            nearest_idx = max(0, min(nearest_idx, len(classes)-1))
                            score_mapping = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 50}
                            estimated_score = score_mapping.get(classes[nearest_idx], 75)
                            display_score = float(estimated_score)
                            st.info("⚠️ หมายเหตุ: โมเดลนี้ทำนายจากระดับเกรดที่แปลงเป็นตัวเลข ระบบจึงประมาณค่าเป็นช่วงคะแนนให้")
                        else:
                            display_score = raw_score

                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; text-align: center; border-radius: 15px;">
                            <h1 style="color: white; margin: 0; font-size: 96px; font-weight: bold;">{display_score:.2f}</h1>
                            <p style="color: white; margin: 10px 0; font-size: 18px;">คะแนนที่ทำนาย (Linear Regression)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("📈 **Linear Regression** ทำนายค่าตัวเลขต่อเนื่อง (คะแนน)")
                        if hasattr(models[model_key], 'coef_'):
                            st.markdown("### 📊 สัมประสิทธิ์ (Coefficients)")
                            coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': models[model_key].coef_})
                            st.dataframe(coef_df, width='stretch')

                    # Bottom Metrics Cards
                    st.markdown("---")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; text-align: center;">
                            <p style="color: white; margin: 0; font-size: 14px;">⏱️ เวลาที่ใช้</p>
                            <h3 style="color: white; margin: 10px 0;">{exec_time:.4f}</h3>
                            <p style="color: white; margin: 0; font-size: 12px;">วินาที</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 12px; text-align: center;">
                            <p style="color: white; margin: 0; font-size: 14px;">🎯 โมเดล</p>
                            <h3 style="color: white; margin: 10px 0;">{model_choice}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with m_col3:
                        acc_key = "KNN" if model_choice == "K-Nearest Neighbor" else model_choice
                        acc_val = models['info'].get('scores', {}).get(acc_key, 0.85)
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 12px; text-align: center;">
                            <p style="color: white; margin: 0; font-size: 14px;">📈 Accuracy / R²</p>
                            <h3 style="color: white; margin: 10px 0;">{acc_val:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.success("✅ ทำนายผลสำเร็จ!")
                else:
                    st.error(f"⚠️ ไม่พบโมเดล {model_choice}")
            except Exception as e:
                st.error(f"️ เกิดข้อผิดพลาด: {str(e)}")
        else:
            st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบโฟลเดอร์ models/")

# ============================================
# หน้า 3: เปรียบเทียบโมเดล
# ============================================
elif page == "📊 เปรียบเทียบโมเดล":
    st.markdown("### 🔍 เปรียบเทียบผลลัพธ์จากทุกโมเดล")
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.number_input("🆔 Student ID", min_value=0, value=1001, step=1, key='c_id')
        gender = st.selectbox("👤 Gender", ["Male", "Female"], key='c_g')
        study_time_hours = st.number_input("⏰ Study Time", min_value=0.0, max_value=12.0, value=6.5, step=0.5, key='c_st')
        attendance_percent = st.number_input("📊 Attendance %", min_value=0.0, max_value=100.0, value=95.0, step=1.0, key='c_att')
        sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5, key='c_sl')
        parental_education = st.selectbox("🎓 Parental Education", ["None", "High School", "Bachelors", "Masters", "PhD"], key='c_pe')
    with col2:
        internet_access = st.selectbox("🌐 Internet Access", ["Yes", "No"], key='c_ia')
        extracurricular_activities = st.selectbox("🎯 Extracurricular", ["Yes", "No"], key='c_ea')
        part_time_job = st.selectbox("💼 Part-time Job", ["Yes", "No"], key='c_pt')
        previous_grade = st.number_input("📚 Previous Grade", min_value=0.0, max_value=100.0, value=88.5, step=0.1, key='c_pg')
        final_exam_score = st.number_input("📝 Final Exam Score", min_value=0.0, max_value=100.0, value=92.0, step=0.1, key='c_fe')
    
    if st.button("🔍 เปรียบเทียบทุกโมเดล", width='stretch'):
        input_data = pd.DataFrame({
            'student_id': [student_id], 'gender': [gender], 'study_time_hours': [study_time_hours],
            'attendance_percent': [attendance_percent], 'sleep_hours': [sleep_hours],
            'parental_education': [parental_education], 'internet_access': [internet_access],
            'extracurricular_activities': [extracurricular_activities], 'part_time_job': [part_time_job],
            'previous_grade': [previous_grade], 'final_exam_score': [final_exam_score]
        })
        if models and 'scaler' in models:
            try:
                feature_names = models['info'].get('feature_names', [])
                input_processed = pd.get_dummies(input_data, drop_first=True)
                for col in feature_names:
                    if col not in input_processed.columns: input_processed[col] = 0
                input_processed = input_processed[feature_names]
                scaled_data = models['scaler'].transform(input_processed)
                
                models_to_compare = {'KNN': 'knn', 'Decision Tree': 'dt', 'SVM': 'svm', 'Random Forest': 'rf', 'Linear Regression': 'lr'}
                results = []
                cols = st.columns(5)
                
                for i, (model_name, model_key) in enumerate(models_to_compare.items()):
                    if model_key in models:
                        start_time = time.time()
                        pred = models[model_key].predict(scaled_data)[0]
                        exec_time = time.time() - start_time
                        
                        if models['info'].get('task') == 'classification' and model_name != 'Linear Regression':
                            classes = models['info'].get('classes', ['A', 'B', 'C', 'D', 'F'])
                            res = classes[int(pred)] if int(pred) < len(classes) else str(pred)
                        else:
                            raw_score = pred
                            if raw_score < 5:
                                nearest_idx = int(round(raw_score))
                                nearest_idx = max(0, min(nearest_idx, len(classes)-1))
                                score_mapping = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 50}
                                res = f"{score_mapping.get(classes[nearest_idx], 75):.2f}"
                            else:
                                res = f"{raw_score:.2f}"
                        results.append({'Model': model_name, 'Prediction': res, 'Time': f"{exec_time:.4f}s"})
                
                st.markdown("---")
                st.markdown("### 📊 ผลการทำนายจากทุกโมเดล")
                for i, result in enumerate(results):
                    with cols[i % 5]:
                        pred = result['Prediction']
                        grade_class = f"grade-{pred}" if pred in ['A', 'B', 'C', 'D', 'F'] else ""
                        st.markdown(f"""
                        <div class="model-card {grade_class}" style="text-align: center;">
                            <h4 style="color: white; margin: 0;">{result['Model']}</h4>
                            <h2 style="color: #667eea; margin: 10px 0;">{pred}</h2>
                            <p style="color: #95a5a6; font-size: 12px;">⏱️ {result['Time']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 📋 ตารางเปรียบเทียบ")
                st.dataframe(pd.DataFrame(results), width='stretch')
            except Exception as e:
                st.error(f"⚠️ เกิดข้อผิดพลาด: {str(e)}")

# ============================================
# หน้า 4: K-Means Clustering (แก้ไขสมบูรณ์)
# ============================================
elif page == "🔵 K-Means Clustering":
    st.markdown("### 🔵 K-Means Clustering - จัดกลุ่มนักเรียน")
    if 'kmeans' in models:
        # ✅ แก้ไข: ย้ายมาไว้ด้านบนสุดของบล็อก เพื่อให้ทั้งปุ่มและส่วนอัพโหลดเรียกใช้ได้
        feature_names = models['info'].get('feature_names', [])
        
        st.markdown("กรอกข้อมูลหลักเพื่อประเมินการจัดกลุ่ม (ระบบจะเติมข้อมูลอื่นๆ ให้โดยอัตโนมัติ)")
        col1, col2 = st.columns(2)
        with col1:
            study_time = st.number_input("⏰ Study Time (ชม./วัน)", min_value=0.0, max_value=12.0, value=4.0, step=0.5, key='km_st')
            attendance = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0, key='km_att')
            sleep = st.number_input("😴 Sleep Hours (ชม./วัน)", min_value=0.0, max_value=12.0, value=7.0, step=0.5, key='km_sl')
        with col2:
            prev_grade = st.number_input("📚 Previous Grade", min_value=0.0, max_value=100.0, value=75.0, step=0.1, key='km_pg')
            final_exam = st.number_input("📝 Final Exam Score", min_value=0.0, max_value=100.0, value=80.0, step=0.1, key='km_fe')
            st.markdown("### 🎯 จำนวน Cluster")
            n_clusters = st.slider("เลือกจำนวนกลุ่ม", 2, 5, 3)
        
        if st.button("🔵 จัดกลุ่มข้อมูล", width='stretch'):
            with st.spinner('กำลังประมวลผล...'):
                input_dict = {
                    'student_id': [1],
                    'gender': ['Male'],
                    'study_time_hours': [study_time],
                    'attendance_percent': [attendance],
                    'sleep_hours': [sleep],
                    'parental_education': ['Bachelors'],
                    'internet_access': ['Yes'],
                    'extracurricular_activities': ['Yes'],
                    'part_time_job': ['No'],
                    'previous_grade': [prev_grade],
                    'final_exam_score': [final_exam]
                }
                input_df = pd.DataFrame(input_dict)
                input_processed = pd.get_dummies(input_df, drop_first=True)
                
                for col in feature_names:
                    if col not in input_processed.columns:
                        input_processed[col] = 0
                input_processed = input_processed[feature_names]
                
                cluster = int(models['kmeans'].predict(input_processed)[0])
            
            st.markdown("---")
            st.markdown("### 🎯 ผลการจัดกลุ่ม")
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; text-align: center; border-radius: 15px;">
                <h1 style="color: white; margin: 0; font-size: 72px;">Cluster {cluster}</h1>
                <p style="color: white; margin: 10px 0; font-size: 18px;">นักเรียนที่มีลักษณะนี้อยู่ในกลุ่มที่ {cluster + 1}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if hasattr(models['kmeans'], 'cluster_centers_'):
                st.markdown("### 📊 ลักษณะค่าเฉลี่ยของแต่ละ Cluster")
                
                centers_df = pd.DataFrame(
                    models['kmeans'].cluster_centers_, 
                    columns=feature_names
                )
                
                numeric_cols = ['study_time_hours', 'attendance_percent', 'sleep_hours', 
                               'previous_grade', 'final_exam_score']
                # กรองเฉพาะคอลัมน์ที่มีอยู่จริง
                available_numeric_cols = [col for col in numeric_cols if col in centers_df.columns]
                
                if available_numeric_cols:
                    display_centers = centers_df[available_numeric_cols].copy()
                    display_centers.columns = ['Study Time', 'Attendance', 'Sleep', 
                                              'Previous Grade', 'Final Exam'][:len(available_numeric_cols)]
                    st.dataframe(display_centers, width='stretch')
                
                # ✅ แก้ไข: ใช้ centers_df (ชื่อเดิม) แทน centers (ชื่อที่เปลี่ยนแล้ว) เพื่อป้องกัน KeyError
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    if 'study_time_hours' in centers_df.columns and 'attendance_percent' in centers_df.columns:
                        axes[0].scatter(centers_df['study_time_hours'], 
                                       centers_df['attendance_percent'], 
                                       c=range(len(centers_df)), s=200, cmap='viridis')
                        axes[0].set_xlabel('Study Time (hours)')
                        axes[0].set_ylabel('Attendance (%)')
                        axes[0].set_title('Study Time vs Attendance')
                    
                    if 'previous_grade' in centers_df.columns and 'final_exam_score' in centers_df.columns:
                        axes[1].scatter(centers_df['previous_grade'], 
                                       centers_df['final_exam_score'], 
                                       c=range(len(centers_df)), s=200, cmap='viridis')
                        axes[1].set_xlabel('Previous Grade')
                        axes[1].set_ylabel('Final Exam Score')
                        axes[1].set_title('Previous vs Final Grade')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถแสดงกราฟได้: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 📁 อัพโหลดไฟล์เพื่อจัดกลุ่มทั้งชุด")
        uploaded_file = st.file_uploader("เลือกไฟล์ Excel/CSV", type=['xlsx', 'csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.dataframe(df.head(), width='stretch')
            
            target_col = models['info'].get('target_column', df.columns[-1])
            X_upload = df.drop(columns=[target_col], errors='ignore')
            X_upload_processed = pd.get_dummies(X_upload, drop_first=True)
            
            # ✅ แก้ไข: feature_names ถูกประกาศไว้ด้านบนแล้ว จึงเรียกใช้ได้ไม่มี Error
            for col in feature_names:
                if col not in X_upload_processed.columns:
                    X_upload_processed[col] = 0
            X_upload_processed = X_upload_processed[feature_names]
            
            labels = models['kmeans'].predict(X_upload_processed)
            df['Cluster'] = labels
            st.markdown("### 📊 ผลการจัดกลุ่ม")
            st.dataframe(df, width='stretch')
    else:
        st.error("⚠️ ไม่พบไฟล์ kmeans_model.pkl")

# ============================================
# หน้า 5: ข้อมูลโมเดล
# ============================================
elif page == "📈 ข้อมูลโมเดล":
    st.markdown("### 📊 ข้อมูลและประสิทธิภาพของโมเดล")
    if models and 'info' in models:
        info = models['info']
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("📌 Task Type", info.get('task', 'N/A'))
        with col2: st.metric("🎯 Target Column", info.get('target_column', 'N/A'))
        with col3: st.metric("📊 Number of Features", len(info.get('feature_names', [])))
        
        st.markdown("---")
        if info.get('scores'):
            scores_df = pd.DataFrame({'Model': list(info['scores'].keys()), 'Score': list(info['scores'].values())})
            fig, ax = plt.subplots()
            ax.barh(scores_df['Model'], scores_df['Score'], color='#667eea')
            ax.set_xlim([0, 1])
            ax.set_title('Model Performance')
            st.pyplot(fig)
            st.dataframe(scores_df, width='stretch')
        if info.get('classes'):
            st.markdown("### 🎓 Classes")
            st.write(info['classes'])
    else:
        st.error("⚠️ ไม่พบไฟล์ model_info.pkl")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #95a5a6; padding: 20px;'><p>Created with ❤️ for Student Grade Prediction</p></div>", unsafe_allow_html=True)