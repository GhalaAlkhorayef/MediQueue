import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Page Configuration
st.set_page_config(
    page_title="AI Hospital Emergency Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg, .css-1v0mbdj {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .severity-critical {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(192, 57, 43, 0.4);
        animation: pulse 2s infinite;
    }
    
    .severity-emergency {
        background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(230, 126, 34, 0.4);
    }
    
    .severity-urgent {
        background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(243, 156, 18, 0.4);
    }
    
    .severity-semi-urgent {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
    }
    
    .severity-non-urgent {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.4);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    .login-container {
        max-width: 450px;
        margin: 100px auto;
        padding: 40px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .patient-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .patient-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'patients' not in st.session_state:
    st.session_state.patients = []

# Helper Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    users_file = Path('users.json')
    if users_file.exists():
        with open(users_file, 'r') as f:
            return json.load(f)
    else:
        default_users = {
            'admin': {
                'password': hash_password('admin123'),
                'role': 'admin',
                'name': 'System Administrator'
            },
            'doctor1': {
                'password': hash_password('doctor123'),
                'role': 'doctor',
                'name': 'Dr. Sarah Johnson'
            },
            'patient1': {
                'password': hash_password('patient123'),
                'role': 'patient',
                'name': 'John Doe'
            }
        }
        with open(users_file, 'w') as f:
            json.dump(default_users, f, indent=4)
        return default_users

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=4)

def load_patients():
    patients_file = Path('patients.csv')
    if patients_file.exists():
        return pd.read_csv(patients_file).to_dict('records')
    return []

def save_patients(patients):
    df = pd.DataFrame(patients)
    df.to_csv('patients.csv', index=False)

def load_model_artifacts():
    try:
        model = joblib.load('triage_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        severity_mapping = joblib.load('severity_mapping.pkl')
        return model, scaler, label_encoders, feature_columns, severity_mapping
    except:
        return None, None, None, None, None

def predict_severity(patient_data, model, scaler, label_encoders, feature_columns):
    input_df = pd.DataFrame([patient_data])
    
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col + '_encoded'] = le.transform(input_df[col])
            except:
                input_df[col + '_encoded'] = 0
    
    X = input_df[feature_columns]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    severity_labels = {1: 'Non-Urgent', 2: 'Semi-Urgent', 3: 'Urgent', 4: 'Emergency', 5: 'Critical'}
    return severity_labels[prediction], proba, prediction

# Login Page
def login_page():
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #667eea;'>🏥</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>AI Hospital Triage</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #7f8c8d;'>Emergency Department Management System</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔐 Login", use_container_width=True):
                users = load_users()
                if username in users and users[username]['password'] == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.user_role = users[username]['role']
                    st.session_state.username = username
                    st.session_state.name = users[username]['name']
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        with col_btn2:
            if st.button("📝 Register", use_container_width=True):
                st.session_state.show_register = True
                st.rerun()
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("**Demo Credentials:**\n\n"
                "👨‍⚕️ Doctor: doctor1 / doctor123\n\n"
                "🧑‍💼 Admin: admin / admin123\n\n"
                "🧑‍🦱 Patient: patient1 / patient123")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Patient Interface
def patient_interface():
    st.title("🏥 Emergency Triage Assessment")
    st.markdown(f"### Welcome, {st.session_state.name}")
    
    model, scaler, label_encoders, feature_columns, severity_mapping = load_model_artifacts()
    
    if model is None:
        st.error("⚠️ Model files not found. Please upload the trained model files.")
        return
    
    with st.form("triage_form"):
        st.markdown("### 📋 Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            chief_complaint = st.selectbox("Chief Complaint", [
                'Chest Pain', 'Difficulty Breathing', 'Abdominal Pain',
                'Headache', 'Fever', 'Injury', 'Weakness', 'Dizziness'
            ])
        
        with col2:
            arrival_mode = st.selectbox("Arrival Mode", ["Walk-in", "Ambulance", "Police"])
            consciousness_level = st.selectbox("Consciousness Level", ["Alert", "Drowsy", "Unresponsive"])
            pain_level = st.slider("Pain Level (0-10)", 0, 10, 5)
        
        st.markdown("### 🩺 Vital Signs")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
            temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
        
        with col4:
            bp_systolic = st.number_input("Blood Pressure Systolic", min_value=80, max_value=220, value=120)
            respiratory_rate = st.number_input("Respiratory Rate", min_value=8, max_value=40, value=16)
        
        with col5:
            bp_diastolic = st.number_input("Blood Pressure Diastolic", min_value=40, max_value=140, value=80)
            oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=98)
        
        submitted = st.form_submit_button("🔍 Analyze Severity", use_container_width=True)
        
        if submitted:
            patient_data = {
                'age': age,
                'gender': gender,
                'heart_rate': heart_rate,
                'blood_pressure_systolic': bp_systolic,
                'blood_pressure_diastolic': bp_diastolic,
                'temperature': temperature,
                'respiratory_rate': respiratory_rate,
                'oxygen_saturation': oxygen_saturation,
                'pain_level': pain_level,
                'consciousness_level': consciousness_level,
                'chief_complaint': chief_complaint,
                'arrival_mode': arrival_mode
            }
            
            severity, proba, severity_code = predict_severity(patient_data, model, scaler, label_encoders, feature_columns)
            
            patient_record = {
                **patient_data,
                'patient_id': f'P{len(st.session_state.patients) + 1:05d}',
                'severity': severity,
                'severity_code': int(severity_code),
                'confidence': float(max(proba) * 100),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'Waiting',
                'doctor_notes': '',
                'username': st.session_state.username
            }
            
            st.session_state.patients.append(patient_record)
            save_patients(st.session_state.patients)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='severity-{severity.lower().replace(' ', '-')}'>"
                       f"⚠️ TRIAGE RESULT: {severity}<br>"
                       f"Confidence: {max(proba)*100:.1f}%"
                       f"</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            recommendations = {
                'Critical': "🚨 **IMMEDIATE MEDICAL ATTENTION REQUIRED**\n\nPlease go to the emergency desk immediately. Medical staff will attend to you right away.",
                'Emergency': "⚠️ **URGENT ATTENTION NEEDED**\n\nPlease inform the staff at the front desk. You will be seen shortly.",
                'Urgent': "🔔 **PROMPT ATTENTION RECOMMENDED**\n\nPlease register at the front desk. Expected wait time: 30-60 minutes.",
                'Semi-Urgent': "ℹ️ **STANDARD PRIORITY**\n\nPlease register and wait in the designated area. Expected wait time: 1-2 hours.",
                'Non-Urgent': "✅ **ROUTINE CARE**\n\nYou may be seen by a general practitioner. Expected wait time: 2-4 hours."
            }
            
            st.info(recommendations.get(severity, "Please consult with medical staff."))
            
            st.success(f"✅ Your triage assessment has been recorded. Patient ID: **{patient_record['patient_id']}**")

# Doctor Dashboard
def doctor_dashboard():
    st.title("👨‍⚕️ Doctor Dashboard")
    st.markdown(f"### Welcome, {st.session_state.name}")
    
    st.session_state.patients = load_patients()
    
    if not st.session_state.patients:
        st.info("📋 No patients in the system yet.")
        return
    
    tab1, tab2, tab3 = st.tabs(["🏥 Patient Queue", "📊 Analytics", "🔍 Search Patient"])
    
    with tab1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ['Critical', 'Emergency', 'Urgent', 'Semi-Urgent', 'Non-Urgent'],
            default=['Critical', 'Emergency', 'Urgent', 'Semi-Urgent', 'Non-Urgent']
        )
        
        filtered_patients = [p for p in st.session_state.patients if p['severity'] in severity_filter]
        filtered_patients.sort(key=lambda x: (x['severity_code'], x['timestamp']), reverse=True)
        
        for idx, patient in enumerate(filtered_patients):
            with st.expander(f"🆔 {patient['patient_id']} - {patient['severity']} - {patient['chief_complaint']}", expanded=(idx < 3)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Patient Information**")
                    st.write(f"Age: {patient['age']}")
                    st.write(f"Gender: {patient['gender']}")
                    st.write(f"Arrival: {patient['arrival_mode']}")
                    st.write(f"Time: {patient['timestamp']}")
                
                with col2:
                    st.markdown("**Vital Signs**")
                    st.write(f"HR: {patient['heart_rate']} bpm")
                    st.write(f"BP: {patient['blood_pressure_systolic']}/{patient['blood_pressure_diastolic']}")
                    st.write(f"Temp: {patient['temperature']}°C")
                    st.write(f"SpO2: {patient['oxygen_saturation']}%")
                
                with col3:
                    st.markdown("**Assessment**")
                    st.write(f"Pain: {patient['pain_level']}/10")
                    st.write(f"Consciousness: {patient['consciousness_level']}")
                    st.write(f"Confidence: {patient['confidence']:.1f}%")
                    st.write(f"Status: {patient.get('status', 'Waiting')}")
                
                st.markdown("---")
                
                new_severity = st.selectbox(f"Override Severity", 
                    ['Critical', 'Emergency', 'Urgent', 'Semi-Urgent', 'Non-Urgent'],
                    index=['Critical', 'Emergency', 'Urgent', 'Semi-Urgent', 'Non-Urgent'].index(patient['severity']),
                    key=f"sev_{idx}")
                
                new_status = st.selectbox(f"Update Status",
                    ['Waiting', 'In Progress', 'Completed', 'Discharged', 'Admitted'],
                    index=['Waiting', 'In Progress', 'Completed', 'Discharged', 'Admitted'].index(patient.get('status', 'Waiting')),
                    key=f"status_{idx}")
                
                notes = st.text_area("Doctor's Notes", value=patient.get('doctor_notes', ''), key=f"notes_{idx}")
                
                if st.button(f"💾 Save Changes", key=f"save_{idx}"):
                    patient['severity'] = new_severity
                    patient['status'] = new_status
                    patient['doctor_notes'] = notes
                    patient['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    patient['updated_by'] = st.session_state.name
                    save_patients(st.session_state.patients)
                    st.success("✅ Patient record updated!")
                    st.rerun()
    
    with tab2:
        df = pd.DataFrame(st.session_state.patients)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            critical = len(df[df['severity'].isin(['Critical', 'Emergency'])])
            st.metric("Critical/Emergency", critical)
        with col3:
            waiting = len(df[df.get('status', 'Waiting') == 'Waiting'])
            st.metric("Waiting", waiting)
        with col4:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            severity_counts = df['severity'].value_counts()
            fig1 = px.pie(values=severity_counts.values, names=severity_counts.index,
                         title='Severity Distribution',
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(df, x='chief_complaint', color='severity',
                         title='Chief Complaints by Severity',
                         color_discrete_map={
                             'Critical': '#c0392b',
                             'Emergency': '#e67e22',
                             'Urgent': '#f39c12',
                             'Semi-Urgent': '#3498db',
                             'Non-Urgent': '#2ecc71'
                         })
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        search_id = st.text_input("🔍 Search by Patient ID")
        if search_id:
            results = [p for p in st.session_state.patients if search_id.lower() in p['patient_id'].lower()]
            if results:
                for patient in results:
                    st.json(patient)
            else:
                st.warning("No patient found with that ID")

# Admin Panel
def admin_panel():
    st.title("🔧 Admin Panel")
    st.markdown(f"### Welcome, {st.session_state.name}")
    
    tab1, tab2, tab3 = st.tabs(["👥 User Management", "📈 System Analytics", "📜 Audit Logs"])
    
    with tab1:
        st.markdown("### Current Users")
        users = load_users()
        
        users_df = pd.DataFrame([
            {'Username': k, 'Name': v['name'], 'Role': v['role']}
            for k, v in users.items()
        ])
        st.dataframe(users_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Add New User")
        
        with st.form("add_user"):
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
            with col2:
                new_name = st.text_input("Full Name")
                new_role = st.selectbox("Role", ["patient", "doctor", "admin"])
            
            if st.form_submit_button("➕ Add User"):
                if new_username in users:
                    st.error("Username already exists!")
                elif new_username and new_password and new_name:
                    users[new_username] = {
                        'password': hash_password(new_password),
                        'role': new_role,
                        'name': new_name
                    }
                    save_users(users)
                    st.success(f"✅ User {new_username} added successfully!")
                    st.rerun()
                else:
                    st.error("Please fill all fields")
        
        st.markdown("---")
        st.markdown("### Remove User")
        user_to_remove = st.selectbox("Select user to remove", list(users.keys()))
        if st.button("🗑️ Remove User"):
            if user_to_remove != 'admin':
                del users[user_to_remove]
                save_users(users)
                st.success(f"✅ User {user_to_remove} removed!")
                st.rerun()
            else:
                st.error("Cannot remove admin user!")
    
    with tab2:
        patients = load_patients()
        users = load_users()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", len(users))
        with col2:
            st.metric("Total Patients", len(patients))
        with col3:
            doctors = sum(1 for u in users.values() if u['role'] == 'doctor')
            st.metric("Doctors", doctors)
        with col4:
            if patients:
                avg_wait = "N/A"
            st.metric("System Status", "🟢 Online")
        
        if patients:
            df = pd.DataFrame(patients)
            
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            fig = px.line(daily_counts, x='date', y='count',
                         title='Daily Patient Volume',
                         labels={'count': 'Number of Patients', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                age_dist = px.histogram(df, x='age', nbins=20,
                                       title='Age Distribution',
                                       labels={'age': 'Age', 'count': 'Frequency'})
                st.plotly_chart(age_dist, use_container_width=True)
            
            with col2:
                gender_dist = df['gender'].value_counts()
                fig_gender = px.pie(values=gender_dist.values, names=gender_dist.index,
                                   title='Gender Distribution')
                st.plotly_chart(fig_gender, use_container_width=True)
    
    with tab3:
        st.markdown("### Recent System Activity")
        if patients:
            df = pd.DataFrame(patients)
            df_sorted = df.sort_values('timestamp', ascending=False)
            
            for _, row in df_sorted.head(20).iterrows():
                st.markdown(f"""
                <div class='patient-card'>
                    <strong>{row['timestamp']}</strong> - Patient {row['patient_id']} 
                    ({row['severity']}) - {row.get('status', 'Waiting')}
                    {f"- Updated by {row.get('updated_by', 'System')}" if 'updated_by' in row else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No activity logs yet")

# Main App Logic
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        # Sidebar
        with st.sidebar:
            st.markdown(f"### 👤 {st.session_state.name}")
            st.markdown(f"**Role:** {st.session_state.user_role.title()}")
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_role = None
                st.session_state.username = None
                st.rerun()
        
        # Route to appropriate interface
        if st.session_state.user_role == 'patient':
            patient_interface()
        elif st.session_state.user_role == 'doctor':
            doctor_dashboard()
        elif st.session_state.user_role == 'admin':
            admin_panel()

if __name__ == "__main__":
    main()