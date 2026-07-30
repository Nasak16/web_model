import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stNumberInput>div>div>input {
        background-color: #262730;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_dir = 'models'
    
    if os.path.exists(model_dir):
        files = {
            'knn': 'knn_model.pkl',
            'dt': 'decision_tree_model.pkl',
            'svm': 'svm_model.pkl',
            'rf': 'random_forest_model.pkl',
            'lr': 'linear_regression_model.pkl',
            'scaler': 'scaler.pkl',
            'info': 'model_info.pkl'
        }
        
        for key, filename in files.items():
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                models[key] = joblib.load(path)
    
    return models

models = load_models()

# Title
st.title("🎓 Student Grade Prediction System")
st.markdown("### ทำนายเกรดนักเรียนจากปัจจัยต่างๆ")

# Sidebar
st.sidebar.title("⚙️ การตั้งค่า")
model_choice = st.sidebar.selectbox(
    "เลือกโมเดล",
    ["K-Nearest Neighbor", "Decision Tree", "SVM", "Random Forest", "Linear Regression"]
)

# Main form
st.markdown("### 📝 กรอกข้อมูลนักเรียน")

col1, col2 = st.columns(2)

with col1:
    student_id = st.number_input(" Student ID", min_value=0, value=1, step=1)
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    study_time_hours = st.number_input("⏰ Study Time (hours/day)", min_value=0.0, max_value=12.0, value=4.0, step=0.5)
    attendance_percent = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    sleep_hours = st.number_input("😴 Sleep Hours (per day)", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
    parental_education = st.selectbox("🎓 Parental Education", ["None", "High School", "Bachelors", "Masters", "PhD"])

with col2:
    internet_access = st.selectbox("🌐 Internet Access", ["Yes", "No"])
    extracurricular_activities = st.selectbox("🎯 Extracurricular Activities", ["Yes", "No"])
    part_time_job = st.selectbox("💼 Part-time Job", ["Yes", "No"])
    previous_grade = st.number_input(" Previous Grade", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    final_exam_score = st.number_input(" Final Exam Score", min_value=0.0, max_value=100.0, value=80.0, step=0.1)

# Prediction button
if st.button(" ทำนายเกรด", use_container_width=True):
    
    # Prepare input data
    input_data = {
        'student_id': student_id,
        'gender': gender,
        'study_time_hours': study_time_hours,
        'attendance_percent': attendance_percent,
        'sleep_hours': sleep_hours,
        'parental_education': parental_education,
        'internet_access': internet_access,
        'extracurricular_activities': extracurricular_activities,
        'part_time_job': part_time_job,
        'previous_grade': previous_grade,
        'final_exam_score': final_exam_score
    }
    
    # Display input summary
    st.markdown("---")
    st.markdown("### 📋 สรุปข้อมูลที่กรอก")
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df, use_container_width=True)
    
    # Make prediction
    if models and 'scaler' in models and 'info' in models:
        try:
            # Get feature names from model info
            feature_names = models['info'].get('feature_names', [])
            
            # Create DataFrame with correct features
            # Convert categorical to dummy variables
            input_df_processed = pd.get_dummies(input_df, drop_first=True)
            
            # Ensure all required columns exist
            for col in feature_names:
                if col not in input_df_processed.columns:
                    input_df_processed[col] = 0
            
            # Reorder columns to match training data
            input_df_processed = input_df_processed[feature_names]
            
            # Scale the data
            scaled_data = models['scaler'].transform(input_df_processed)
            
            # Select model based on choice
            model_map = {
                "K-Nearest Neighbor": 'knn',
                "Decision Tree": 'dt',
                "SVM": 'svm',
                "Random Forest": 'rf',
                "Linear Regression": 'lr'
            }
            
            model_key = model_map.get(model_choice, 'rf')
            
            if model_key in models:
                prediction = models[model_key].predict(scaled_data)
                
                # Display result
                st.markdown("---")
                st.markdown("### 🎯 ผลการทำนาย")
                
                if models['info'].get('task') == 'classification':
                    classes = models['info'].get('classes', ['A', 'B', 'C', 'D', 'F'])
                    predicted_grade = classes[int(prediction[0])] if prediction[0] < len(classes) else prediction[0]
                    
                    # Color code the grade
                    grade_color = {
                        'A': '#4CAF50',
                        'B': '#8BC34A',
                        'C': '#FFC107',
                        'D': '#FF9800',
                        'F': '#F44336'
                    }.get(str(predicted_grade), '#9E9E9E')
                    
                    st.markdown(f"""
                    <div class="metric-card" style="background: {grade_color}; text-align: center; padding: 30px;">
                        <h2 style="color: white; margin: 0;">เกรดที่預測: {predicted_grade}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; padding: 30px;">
                        <h2 style="color: white; margin: 0;">คะแนนที่預測: {prediction[0]:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("✅ ทำนายผลสำเร็จ!")
                
            else:
                st.error(f"️ ไม่พบโมเดล {model_choice}")
                
        except Exception as e:
            st.error(f"⚠️ เกิดข้อผิดพลาด: {str(e)}")
            st.info(" ตรวจสอบว่าไฟล์โมเดลถูกต้องและครบถ้วน")
    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาอัพโหลดไฟล์ .pkl ในโฟลเดอร์ models/")
        st.info("""
        **ไฟล์ที่ต้องการ:**
        - knn_model.pkl
        - decision_tree_model.pkl
        - svm_model.pkl
        - random_forest_model.pkl
        - linear_regression_model.pkl
        - scaler.pkl
        - model_info.pkl
        """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #95a5a6;'><p>Created with ❤️ for Student Grade Prediction</p></div>", unsafe_allow_html=True)