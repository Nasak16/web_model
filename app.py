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
    page_icon="",
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
    .developer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .model-card {
        background: #1e1e2e;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .grade-A { background: linear-gradient(135deg, #4CAF50, #8BC34A); }
    .grade-B { background: linear-gradient(135deg, #2196F3, #03A9F4); }
    .grade-C { background: linear-gradient(135deg, #FFC107, #FF9800); }
    .grade-D { background: linear-gradient(135deg, #FF5722, #F44336); }
    .grade-F { background: linear-gradient(135deg, #9E9E9E, #607D8B); }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ============================================
# ส่วนข้อมูลผู้พัฒนา (ตาม Requirement)
# ============================================

def show_developer_info():
    st.markdown("### ‍💻 ข้อมูลผู้พัฒนา")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # อัพโหลดรูปหรือใช้รูป default
        uploaded_image = st.file_uploader("อัพโหลดรูปผู้พัฒนา", type=['png', 'jpg', 'jpeg'])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, width=200)
        else:
            st.image("https://via.placeholder.com/200", caption="รูปผู้พัฒนา", width=200)
    
    with col2:
        name = st.text_input("ชื่อ-นามสกุล", "นาย สมชาย ใจดี")
        student_id = st.text_input("รหัสนักศึกษา", "6501234567")
        group = st.text_input("หมู่เรียน", "CS-2A")
        
        st.markdown(f"""
        <div class="developer-card">
            <h2 style="color: white; margin: 0;">{name}</h2>
            <p style="color: white; margin: 10px 0;">รหัส: {student_id}</p>
            <p style="color: white; margin: 10px 0;">หมู่เรียน: {group}</p>
        </div>
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
            'kmeans': 'kmeans_model.pkl',
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
st.title(" Student Grade Prediction System")

# Navigation
page = st.sidebar.selectbox(
    "📌 เลือกหน้า",
    ["👨‍ ข้อมูลผู้พัฒนา", "🔮 ทำนายผล", "📊 เปรียบเทียบโมเดล", 
     "🔵 K-Means Clustering", "📈 ข้อมูลโมเดล"]
)

# ============================================
# หน้า 1: ข้อมูลผู้พัฒนา
# ============================================
if page == "‍💻 ข้อมูลผู้พัฒนา":
    show_developer_info()
    
    st.markdown("---")
    st.markdown("### 📋 เกี่ยวกับโปรเจค")
    st.markdown("""
    **Student Grade Prediction System**
    
    โปรเจคนี้พัฒนาเพื่อทำนายเกรดนักเรียนจากปัจจัยต่างๆ เช่น:
    - ⏰ เวลาเรียน
    -  อัตราการเข้าเรียน
    -  ชั่วโมงการนอน
    - 🎓 การศึกษาของผู้ปกครอง
    - และอื่นๆ
    
    **เทคโนโลยีที่ใช้:**
    - Python
    - Scikit-Learn
    - Streamlit
    - Machine Learning Models
    """)

# ============================================
# หน้า 2: ทำนายผล
# ============================================
elif page == "🔮 ทำนายผล":
    st.markdown("### 🎯 ทำนายเกรดด้วยโมเดลต่างๆ")
    
    model_choice = st.selectbox(
        "เลือกโมเดล",
        ["K-Nearest Neighbor", "Decision Tree", "SVM", "Random Forest", "Linear Regression"]
    )
    
    # Form
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.number_input("🆔 Student ID", min_value=0, value=1, step=1)
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        study_time_hours = st.number_input("⏰ Study Time (hours/day)", min_value=0.0, max_value=12.0, value=4.0, step=0.5)
        attendance_percent = st.number_input(" Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        parental_education = st.selectbox("🎓 Parental Education", ["None", "High School", "Bachelors", "Masters", "PhD"])
    
    with col2:
        internet_access = st.selectbox("🌐 Internet Access", ["Yes", "No"])
        extracurricular_activities = st.selectbox("🎯 Extracurricular", ["Yes", "No"])
        part_time_job = st.selectbox("💼 Part-time Job", ["Yes", "No"])
        previous_grade = st.number_input("📚 Previous Grade", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        final_exam_score = st.number_input("📝 Final Exam Score", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    
    if st.button("🔮 ทำนายผล", use_container_width=True):
        input_data = pd.DataFrame({
            'student_id': [student_id],
            'gender': [gender],
            'study_time_hours': [study_time_hours],
            'attendance_percent': [attendance_percent],
            'sleep_hours': [sleep_hours],
            'parental_education': [parental_education],
            'internet_access': [internet_access],
            'extracurricular_activities': [extracurricular_activities],
            'part_time_job': [part_time_job],
            'previous_grade': [previous_grade],
            'final_exam_score': [final_exam_score]
        })
        
        st.markdown("### 📋 ข้อมูลที่กรอก")
        st.dataframe(input_data, use_container_width=True)
        
        if models and 'scaler' in models:
            try:
                # Process data
                feature_names = models['info'].get('feature_names', [])
                input_processed = pd.get_dummies(input_data, drop_first=True)
                
                for col in feature_names:
                    if col not in input_processed.columns:
                        input_processed[col] = 0
                
                input_processed = input_processed[feature_names]
                scaled_data = models['scaler'].transform(input_processed)
                
                # Model mapping
                model_map = {
                    "K-Nearest Neighbor": 'knn',
                    "Decision Tree": 'dt',
                    "SVM": 'svm',
                    "Random Forest": 'rf',
                    "Linear Regression": 'lr'
                }
                
                model_key = model_map[model_choice]
                
                if model_key in models:
                    start_time = time.time()
                    prediction = models[model_key].predict(scaled_data)
                    exec_time = time.time() - start_time
                    
                    st.markdown("---")
                    st.markdown(f"### 🎯 ผลการทำนายจาก {model_choice}")
                    
                    # แสดงผลตามประเภทโมเดล
                    if model_choice == "K-Nearest Neighbor":
                        # KNN: แสดง neighbors
                        classes = models['info'].get('classes', ['A', 'B', 'C', 'D', 'F'])
                        grade = classes[int(prediction[0])] if int(prediction[0]) < len(classes) else str(prediction[0])
                        
                        st.markdown(f"""
                        <div class="metric-card grade-{grade}" style="padding: 40px; text-align: center;">
                            <h1 style="color: white; margin: 0; font-size: 72px;">{grade}</h1>
                            <p style="color: white; margin: 10px 0;">เกรดที่预测 (KNN)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("💡 **KNN** ทำนายโดยดูจากเพื่อนบ้านที่ใกล้ที่สุด (K neighbors)")
                        
                        # แสดง K value
                        if hasattr(models[model_key], 'n_neighbors'):
                            st.metric("จำนวน K (Neighbors)", models[model_key].n_neighbors)
                    
                    elif model_choice == "Decision Tree":
                        # Decision Tree: แสดง feature importance
                        classes = models['info'].get('classes', ['A', 'B', 'C', 'D', 'F'])
                        grade = classes[int(prediction[0])] if int(prediction[0]) < len(classes) else str(prediction[0])
                        
                        st.markdown(f"""
                        <div class="metric-card grade-{grade}" style="padding: 40px; text-align: center;">
                            <h1 style="color: white; margin: 0; font-size: 72px;">{grade}</h1>
                            <p style="color: white; margin: 10px 0;">เกรดที่预测 (Decision Tree)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("🌳 **Decision Tree** ตัดสินใจตามเงื่อนไขของ features")
                        
                        # Feature Importance
                        if hasattr(models[model_key], 'feature_importances_'):
                            st.markdown("### 📊 ความสำคัญของ Features")
                            imp_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': models[model_key].feature_importances_
                            }).sort_values('Importance', ascending=True)
                            st.bar_chart(imp_df.set_index('Feature'))
                    
                    elif model_choice == "SVM":
                        # SVM: แสดง confidence
                        classes = models['info'].get('classes', ['A', 'B', 'C', 'D', 'F'])
                        grade = classes[int(prediction[0])] if int(prediction[0]) < len(classes) else str(prediction[0])
                        
                        st.markdown(f"""
                        <div class="metric-card grade-{grade}" style="padding: 40px; text-align: center;">
                            <h1 style="color: white; margin: 0; font-size: 72px;">{grade}</h1>
                            <p style="color: white; margin: 10px 0;">เกรดที่预测 (SVM)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("⚡ **SVM** หา hyperplane ที่แยก classes ได้ดีที่สุด")
                        
                        # Confidence score
                        if hasattr(models[model_key], 'decision_function'):
                            confidence = models[model_key].decision_function(scaled_data)[0]
                            st.metric("Confidence Score", f"{confidence[0]:.4f}")
                    
                    elif model_choice == "Random Forest":
                        # Random Forest: แสดง voting
                        classes = models['info'].get('classes', ['A', 'B', 'C', 'D', 'F'])
                        grade = classes[int(prediction[0])] if int(prediction[0]) < len(classes) else str(prediction[0])
                        
                        st.markdown(f"""
                        <div class="metric-card grade-{grade}" style="padding: 40px; text-align: center;">
                            <h1 style="color: white; margin: 0; font-size: 72px;">{grade}</h1>
                            <p style="color: white; margin: 10px 0;">เกรดที่预测 (Random Forest)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("🌲 **Random Forest** โหวตจาก trees หลายต้น")
                        
                        # Number of trees
                        if hasattr(models[model_key], 'n_estimators'):
                            st.metric("จำนวน Trees", models[model_key].n_estimators)
                        
                        # Feature Importance
                        if hasattr(models[model_key], 'feature_importances_'):
                            st.markdown("### 📊 Feature Importance")
                            imp_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': models[model_key].feature_importances_
                            }).sort_values('Importance', ascending=True)
                            st.bar_chart(imp_df.set_index('Feature'))
                    
                    elif model_choice == "Linear Regression":
                        # Regression: แสดงค่าที่ทำนาย
                        st.markdown(f"""
                        <div class="metric-card" style="padding: 40px; text-align: center;">
                            <h1 style="color: white; margin: 0; font-size: 72px;">{prediction[0]:.2f}</h1>
                            <p style="color: white; margin: 10px 0;">คะแนนที่预测 (Regression)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("📈 **Linear Regression** ทำนายค่าต่อเนื่อง")
                        
                        # Coefficients
                        if hasattr(models[model_key], 'coef_'):
                            st.markdown("### 📊 สัมประสิทธิ์ (Coefficients)")
                            coef_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Coefficient': models[model_key].coef_
                            })
                            st.dataframe(coef_df)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ เวลาที่ใช้", f"{exec_time:.4f} วินาที")
                    with col2:
                        st.metric("🎯 โมเดล", model_choice)
                    with col3:
                        if models['info'].get('scores', {}).get(model_choice):
                            st.metric("📈 Accuracy", f"{models['info']['scores'][model_choice]:.2%}")
                    
                    st.success("✅ ทำนายผลสำเร็จ!")
                    
                else:
                    st.error(f"⚠️ ไม่พบโมเดล {model_choice}")
                    
            except Exception as e:
                st.error(f"️ เกิดข้อผิดพลาด: {str(e)}")
        else:
            st.error("⚠️ ไม่พบไฟล์โมเดล")

# ============================================
# หน้า 3: เปรียบเทียบโมเดล
# ============================================
elif page == "📊 เปรียบเทียบโมเดล":
    st.markdown("### 🔍 เปรียบเทียบผลลัพธ์จากทุกโมเดล")
    
    # Form (เหมือนด้านบน)
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.number_input("🆔 Student ID", min_value=0, value=1, step=1, key='compare_id')
        gender = st.selectbox("👤 Gender", ["Male", "Female"], key='compare_gender')
        study_time_hours = st.number_input("⏰ Study Time", min_value=0.0, max_value=12.0, value=4.0, step=0.5, key='compare_study')
        attendance_percent = st.number_input("📊 Attendance %", min_value=0.0, max_value=100.0, value=85.0, step=1.0, key='compare_attend')
        sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5, key='compare_sleep')
        parental_education = st.selectbox("🎓 Parental Education", ["None", "High School", "Bachelors", "Masters", "PhD"], key='compare_parent')
    
    with col2:
        internet_access = st.selectbox("🌐 Internet Access", ["Yes", "No"], key='compare_internet')
        extracurricular_activities = st.selectbox(" Extracurricular", ["Yes", "No"], key='compare_extra')
        part_time_job = st.selectbox("💼 Part-time Job", ["Yes", "No"], key='compare_job')
        previous_grade = st.number_input("📚 Previous Grade", min_value=0.0, max_value=100.0, value=75.0, step=0.1, key='compare_prev')
        final_exam_score = st.number_input("📝 Final Exam Score", min_value=0.0, max_value=100.0, value=80.0, step=0.1, key='compare_final')
    
    if st.button("🔍 เปรียบเทียบทุกโมเดล", use_container_width=True):
        input_data = pd.DataFrame({
            'student_id': [student_id],
            'gender': [gender],
            'study_time_hours': [study_time_hours],
            'attendance_percent': [attendance_percent],
            'sleep_hours': [sleep_hours],
            'parental_education': [parental_education],
            'internet_access': [internet_access],
            'extracurricular_activities': [extracurricular_activities],
            'part_time_job': [part_time_job],
            'previous_grade': [previous_grade],
            'final_exam_score': [final_exam_score]
        })
        
        if models and 'scaler' in models:
            try:
                feature_names = models['info'].get('feature_names', [])
                input_processed = pd.get_dummies(input_data, drop_first=True)
                
                for col in feature_names:
                    if col not in input_processed.columns:
                        input_processed[col] = 0
                
                input_processed = input_processed[feature_names]
                scaled_data = models['scaler'].transform(input_processed)
                
                models_to_compare = {
                    'KNN': 'knn',
                    'Decision Tree': 'dt',
                    'SVM': 'svm',
                    'Random Forest': 'rf',
                    'Linear Regression': 'lr'
                }
                
                results = []
                
                st.markdown("---")
                st.markdown("### 📊 ผลการทำนายจากทุกโมเดล")
                
                cols = st.columns(5)
                
                for i, (model_name, model_key) in enumerate(models_to_compare.items()):
                    if model_key in models:
                        start_time = time.time()
                        prediction = models[model_key].predict(scaled_data)[0]
                        exec_time = time.time() - start_time
                        
                        if models['info'].get('task') == 'classification':
                            classes = models['info'].get('classes', ['A', 'B', 'C', 'D', 'F'])
                            grade = classes[int(prediction)] if int(prediction) < len(classes) else str(prediction)
                            results.append({'Model': model_name, 'Prediction': grade, 'Time': exec_time})
                        else:
                            results.append({'Model': model_name, 'Prediction': f"{prediction:.2f}", 'Time': exec_time})
                
                # Display results
                for i, result in enumerate(results):
                    with cols[i % 5]:
                        pred = result['Prediction']
                        grade_class = f"grade-{pred}" if pred in ['A', 'B', 'C', 'D', 'F'] else ""
                        
                        st.markdown(f"""
                        <div class="model-card {grade_class}" style="text-align: center;">
                            <h4 style="color: white; margin: 0;">{result['Model']}</h4>
                            <h2 style="color: #667eea; margin: 10px 0;">{pred}</h2>
                            <p style="color: #95a5a6; font-size: 12px;">⏱️ {result['Time']:.4f}s</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Comparison table
                st.markdown("---")
                st.markdown("### 📋 ตารางเปรียบเทียบ")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Chart
                st.markdown("### 📊 กราฟเปรียบเทียบ")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    ax.barh(results_df['Model'], results_df['Time'], color='#667eea')
                    ax.set_xlabel('เวลา (วินาที)')
                    ax.set_title('⏱️ เวลาที่ใช้ในการทำนาย')
                    st.pyplot(fig)
                
                with col2:
                    if models['info'].get('scores'):
                        scores = {k: v for k, v in models['info']['scores'].items() if k in results_df['Model'].values}
                        fig, ax = plt.subplots()
                        ax.barh(list(scores.keys()), list(scores.values()), color='#4CAF50')
                        ax.set_xlim([0, 1])
                        ax.set_xlabel('Score')
                        ax.set_title('🎯 ความแม่นยำของโมเดล')
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"⚠️ เกิดข้อผิดพลาด: {str(e)}")

# ============================================
# หน้า 4: K-Means Clustering
# ============================================
elif page == "🔵 K-Means Clustering":
    st.markdown("### 🔵 K-Means Clustering - จัดกลุ่มนักเรียน")
    
    if 'kmeans' in models:
        col1, col2 = st.columns(2)
        
        with col1:
            study_time = st.number_input("⏰ Study Time", min_value=0.0, max_value=12.0, value=4.0, step=0.5, key='km_study')
            attendance = st.number_input("📊 Attendance %", min_value=0.0, max_value=100.0, value=85.0, step=1.0, key='km_attend')
            sleep = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5, key='km_sleep')
            prev_grade = st.number_input("📚 Previous Grade", min_value=0.0, max_value=100.0, value=75.0, step=0.1, key='km_prev')
        
        with col2:
            final_exam = st.number_input("📝 Final Exam", min_value=0.0, max_value=100.0, value=80.0, step=0.1, key='km_final')
            st.markdown("### 🎯 จำนวน Cluster")
            n_clusters = st.slider("เลือกจำนวนกลุ่ม", 2, 5, 3)
        
        if st.button("🔵 จัดกลุ่มข้อมูล", use_container_width=True):
            input_data = np.array([[study_time, attendance, sleep, prev_grade, final_exam]])
            
            kmeans_model = models['kmeans']
            cluster = kmeans_model.predict(input_data)[0]
            
            st.markdown("---")
            st.markdown("### 🎯 ผลการจัดกลุ่ม")
            
            st.markdown(f"""
            <div class="metric-card" style="padding: 40px; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 72px;">Cluster {cluster}</h1>
                <p style="color: white; margin: 10px 0;">นักเรียนนี้อยู่ในกลุ่มที่ {cluster + 1}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cluster characteristics
            st.markdown("### 📊 ลักษณะของแต่ละ Cluster")
            
            if hasattr(kmeans_model, 'cluster_centers_'):
                centers = pd.DataFrame(
                    kmeans_model.cluster_centers_,
                    columns=['Study Time', 'Attendance', 'Sleep', 'Previous Grade', 'Final Exam']
                )
                st.dataframe(centers)
                
                # Visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                axes[0].scatter(centers['Study Time'], centers['Attendance'], 
                               c=range(len(centers)), s=200, cmap='viridis')
                axes[0].set_xlabel('Study Time')
                axes[0].set_ylabel('Attendance')
                axes[0].set_title('Study Time vs Attendance')
                
                axes[1].scatter(centers['Previous Grade'], centers['Final Exam'],
                               c=range(len(centers)), s=200, cmap='viridis')
                axes[1].set_xlabel('Previous Grade')
                axes[1].set_ylabel('Final Exam')
                axes[1].set_title('Previous vs Final Grade')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Upload data
        st.markdown("---")
        st.markdown("### 📁 อัพโหลดไฟล์เพื่อจัดกลุ่ม")
        uploaded_file = st.file_uploader("เลือกไฟล์ Excel/CSV", type=['xlsx', 'csv'])
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.markdown("### 📊 ข้อมูลที่อัพโหลด")
            st.dataframe(df.head())
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                X = df[numeric_cols].values
                labels = kmeans_model.predict(X)
                
                df['Cluster'] = labels
                
                st.markdown("###  ผลการจัดกลุ่ม")
                st.dataframe(df)
                
                if len(numeric_cols) >= 2:
                    fig, ax = plt.subplots()
                    scatter = ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                                        c=labels, cmap='viridis', alpha=0.6)
                    ax.scatter(kmeans_model.cluster_centers_[:, 0], 
                              kmeans_model.cluster_centers_[:, 1],
                              c='red', s=200, marker='X', label='Centroids')
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                    ax.legend()
                    st.pyplot(fig)
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
        
        with col1:
            st.metric(" Task Type", info.get('task', 'N/A'))
        with col2:
            st.metric("🎯 Target Column", info.get('target_column', 'N/A'))
        with col3:
            st.metric("📊 Number of Features", len(info.get('feature_names', [])))
        
        st.markdown("---")
        st.markdown("### 📈 คะแนนความแม่นยำ")
        
        if info.get('scores'):
            scores_df = pd.DataFrame({
                'Model': list(info['scores'].keys()),
                'Score': list(info['scores'].values())
            })
            
            fig, ax = plt.subplots()
            ax.barh(scores_df['Model'], scores_df['Score'], color='#667eea')
            ax.set_xlim([0, 1])
            ax.set_xlabel('Score')
            ax.set_title('Model Performance')
            st.pyplot(fig)
            
            st.dataframe(scores_df, use_container_width=True)
        
        if info.get('classes'):
            st.markdown("### 🎓 Classes")
            st.write(info['classes'])
        
        st.markdown("---")
        st.markdown("###  Feature Names")
        st.write(info.get('feature_names', []))
    else:
        st.error("⚠️ ไม่พบไฟล์ model_info.pkl")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #95a5a6;'><p>Created with ❤️ for Student Grade Prediction</p></div>", unsafe_allow_html=True)