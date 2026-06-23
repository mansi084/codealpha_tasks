import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# FastAPI endpoint host config
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Clinical Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Theme Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
        color: #212529;
    }
    .sidebar .sidebar-content {
        background-color: #0D5C75;
    }
    .stButton>button {
        background-color: #1982A1;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0D5C75;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #1982A1;
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #0D5C75;
        font-family: 'Outfit', sans-serif;
    }
    .glass-panel {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(13, 92, 117, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

st.title("👨‍⚕️ Clinical Decision Support System (CDSS)")
st.caption("AI-Powered Disease Risk Diagnostics, Explainable AI (SHAP/LIME), and Preventive Guidelines")

# Sidebar navigation
st.sidebar.title("Navigation")
tab_selection = st.sidebar.radio(
    "Select System Tab",
    ["Patient Registration & Prediction", "Explainable AI Insights", "Clinical Analytics Dashboard", "RAG Clinical Chat Assistant", "Batch Processing"]
)

# Shared patient state in session to link intake results between Tab 1 and Tab 2
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None
if "patients_list" not in st.session_state:
    st.session_state.patients_list = []

# Refresh patient lists
def fetch_patients():
    try:
        r = requests.get(f"{API_URL}/patients")
        if r.status_code == 200:
            st.session_state.patients_list = r.json()
    except Exception:
        pass

fetch_patients()

# ----------------- TAB 1: INDIVIDUAL DIAGNOSTICS -----------------
if tab_selection == "Patient Registration & Prediction":
    st.subheader("Patient Vitals Registration & Risk Ingestion")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.write("#### Clinical Intake Form")
        
        name = st.text_input("Patient Full Name", value="Jane Doe")
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=52)
        gender = st.selectbox("Biological Sex", ["Female", "Male"])
        gender_int = 1 if gender == "Male" else 0
        
        st.write("**Vitals & Measurements**")
        w_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=74.0)
        h_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0)
        sys_bp = st.number_input("Systolic BP (mmHg)", min_value=50.0, max_value=250.0, value=138.0)
        dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=30.0, max_value=150.0, value=86.0)
        heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=30.0, max_value=220.0, value=78.0)
        
        st.write("**Laboratory Results**")
        glucose = st.number_input("Glucose level (mg/dL)", min_value=30.0, max_value=600.0, value=112.0)
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=18.0, value=5.9)
        creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=25.0, value=1.0)
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=50.0, max_value=600.0, value=224.0)
        
        st.write("**Symptoms Checklist**")
        chest_pain = st.checkbox("Chest Pain or Discomfort")
        fatigue = st.checkbox("Fatigue or Lethargy")
        freq_urination = st.checkbox("Frequent Urination")
        shortness_of_breath = st.checkbox("Shortness of Breath")
        
        st.write("**Medical History**")
        smoking = st.checkbox("Active Smoker")
        fam_history = st.checkbox("Family History of Diabetes/Heart Disease")
        
        # Calculate derived inputs
        pregnancies = 0
        if gender == "Female":
            pregnancies = st.number_input("Pregnancies Count", min_value=0, max_value=20, value=1)
            
        run_prediction = st.button("Register & Run Diagnostics")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        if run_prediction:
            payload = {
                "name": name, "age": age, "gender": gender_int, "weight_kg": w_kg, "height_cm": h_cm,
                "pregnancies": pregnancies, "systolic_bp": sys_bp, "diastolic_bp": dia_bp,
                "heart_rate": heart_rate, "glucose": glucose, "hba1c": hba1c, "creatinine": creatinine,
                "cholesterol": cholesterol, "chest_pain": 4 if chest_pain else 0, "fatigue": int(fatigue),
                "frequent_urination": int(freq_urination), "shortness_of_breath": int(shortness_of_breath),
                "smoking_status": int(smoking), "family_history": int(fam_history)
            }
            
            with st.spinner("Analyzing patient vitals and computing multi-disease predictions..."):
                try:
                    response = requests.post(f"{API_URL}/predict/all", json=payload)
                    if response.status_code == 200:
                        res = response.json()
                        st.session_state.last_prediction = res
                        fetch_patients()
                        # Pick last patient ID
                        if st.session_state.patients_list:
                            st.session_state.patient_id = st.session_state.patients_list[0]['id']
                    else:
                        st.error(f"Error computing prediction: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
                    
        # Render prediction outcome
        if st.session_state.last_prediction:
            res = st.session_state.last_prediction
            
            st.markdown("### Clinical Risk Assessment Summary")
            
            # Key statistics cards
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.markdown(f"<div class='metric-card'><b>Patient Name</b><br/><h3>{res['patient_name']}</h3></div>", unsafe_allow_html=True)
            with mcol2:
                st.markdown(f"<div class='metric-card'><b>Calculated BMI</b><br/><h3>{res['bmi']:.1f}</h3><small>{res['bmi_category']}</small></div>", unsafe_allow_html=True)
            with mcol3:
                st.markdown(f"<div class='metric-card'><b>BP Category</b><br/><h3>{res['bp_category']}</h3></div>", unsafe_allow_html=True)
            with mcol4:
                st.markdown(f"<div class='metric-card'><b>Composite Health Score</b><br/><h3>{res['composite_health_score']:.0f}/100</h3></div>", unsafe_allow_html=True)
                
            st.write("---")
            st.write("#### Disease Probability & Risk Levels")
            
            # Risk Dials/Bars
            for pred in res['predictions']:
                prob = pred['probability']
                risk_lvl = pred['risk_level']
                recommendation = pred['recommendation']
                
                # Health Bar Colors
                if risk_lvl == "High":
                    bar_color = "red"
                    icon = "🚨"
                elif risk_lvl == "Moderate":
                    bar_color = "orange"
                    icon = "⚠️"
                else:
                    bar_color = "green"
                    icon = "✅"
                    
                st.write(f"**{icon} {pred['disease_name']}** (Risk Level: {risk_lvl})")
                st.progress(prob)
                st.write(f"*Clinical Recommendation:* {recommendation}")
                st.write("")
                
            # PDF Report Download
            if st.session_state.patient_id:
                st.write("---")
                st.write("#### Clinical PDF Documentation")
                try:
                    pdf_url = f"{API_URL}/patient/{st.session_state.patient_id}/pdf"
                    pdf_res = requests.get(pdf_url)
                    if pdf_res.status_code == 200:
                        st.download_button(
                            label="📥 Download AI Health Risk Report (PDF)",
                            data=pdf_res.content,
                            file_name=f"Clinical_Report_{name.replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.write(f"PDF download temporarily unavailable: {e}")
        else:
            st.info("Intake form is ready. Please fill in details and click 'Register & Run Diagnostics' to predict risks.")

# ----------------- TAB 2: EXPLAINABLE AI INSIGHTS -----------------
elif tab_selection == "Explainable AI Insights":
    st.subheader("Explainable AI (SHAP & LIME Tabular Plots)")
    st.write("Select a disease model to see local attribute weight impact on the patient's diagnostic outcome.")
    
    if not st.session_state.last_prediction:
        st.warning("Please complete a prediction first on the 'Patient Registration & Prediction' tab to load explanation charts.")
    else:
        res = st.session_state.last_prediction
        disease_names = [p['disease_name'] for p in res['predictions']]
        selected_dis = st.selectbox("Select Disease Model", disease_names)
        
        # Find prediction dictionary
        pred_dict = [p for p in res['predictions'] if p['disease_name'] == selected_dis][0]
        
        st.write(f"### Feature Importance for {selected_dis}")
        st.write(f"Overall Predicted Likelihood: **{pred_dict['probability']*100:.1f}%** (Risk Category: **{pred_dict['risk_level']}**)")
        
        factors = pred_dict['contributing_factors']
        if not factors:
            st.info("No major feature attributions detected for this patient.")
        else:
            # Build DataFrame
            f_df = pd.DataFrame([{'Feature': f['feature'], 'SHAP Impact': f['impact'], 'Direction': f['description']} for f in factors])
            
            # Generate custom high DPI horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
            colors_list = ['#DC3545' if x > 0 else '#28A745' for x in f_df['SHAP Impact']]
            
            sns.barplot(
                x='SHAP Impact', 
                y='Feature', 
                data=f_df, 
                palette=colors_list, 
                hue='Feature',
                legend=False,
                ax=ax
            )
            ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.set_title("Local Attribution Impacts (Positive increases risk)", fontsize=10, color="#0D5C75", fontweight='bold')
            ax.set_xlabel("Impact Weight", fontsize=8)
            ax.set_ylabel("Patient Feature", fontsize=8)
            ax.tick_params(labelsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Attributions list
            st.write("#### Attribution Summary")
            for idx, row in f_df.iterrows():
                impact_type = "🔴 increases risk" if row['SHAP Impact'] > 0 else "🟢 decreases risk"
                st.write(f"- **{row['Feature']}**: {row['SHAP Impact']:.4f} ({impact_type})")

# ----------------- TAB 3: CLINICAL ANALYTICS -----------------
elif tab_selection == "Clinical Analytics Dashboard":
    st.subheader("Population Health Trends & Database Analytics")
    
    try:
        r = requests.get(f"{API_URL}/model-metrics")
        if r.status_code == 200:
            metrics = r.json()
            
            # Compare model accuracies
            st.write("### Model Performance Comparison Across Diseases")
            
            comparison_rows = []
            for dis, meta in metrics.items():
                for model_name, stats in meta['models'].items():
                    comparison_rows.append({
                        'Disease': dis.replace('_', ' ').title(),
                        'Algorithm': model_name.replace('_', ' ').title(),
                        'Accuracy': stats['accuracy'],
                        'ROC-AUC': stats['roc_auc']
                    })
                    
            c_df = pd.DataFrame(comparison_rows)
            
            # Plotly equivalent barplot
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            sns.barplot(x='Disease', y='Accuracy', hue='Algorithm', data=c_df, ax=ax, palette="mako")
            ax.set_title("Validation Accuracy comparison", fontsize=10, fontweight='bold', color="#0D5C75")
            ax.set_ylim(0.5, 1.05)
            ax.legend(fontsize=6, loc='lower right')
            ax.tick_params(labelsize=8)
            st.pyplot(fig)
            
            # Detailed metrics table
            st.write("#### Model Metrics Log")
            st.dataframe(c_df)
            
        else:
            st.info("Metrics statistics database not found.")
    except Exception as e:
        st.error(f"Could not load analytics metrics: {e}")
        
    st.write("---")
    st.write("#### Population Age & Vitals Distributions")
    if st.session_state.patients_list:
        p_df = pd.DataFrame(st.session_state.patients_list)
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            fig1, ax1 = plt.subplots(figsize=(5, 3), dpi=150)
            sns.histplot(p_df['age'], bins=10, kde=True, ax=ax1, color="#1982A1")
            ax1.set_title("Age distribution of registered patients", fontsize=8, fontweight='bold')
            ax1.tick_params(labelsize=6)
            st.pyplot(fig1)
        with col_an2:
            gender_counts = p_df['gender'].value_counts()
            gender_counts.index = ['Male' if x == 1 else 'Female' for x in gender_counts.index]
            fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=150)
            gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=["#1982A1", "#0D5C75"], ax=ax2, textprops={'fontsize': 6})
            ax2.set_ylabel("")
            ax2.set_title("Gender proportion in database", fontsize=8, fontweight='bold')
            st.pyplot(fig2)
    else:
        st.info("No registered patients in database yet. Predictions will populate this dashboard.")

# ----------------- TAB 4: RAG CLINICAL CHAT ASSISTANT -----------------
elif tab_selection == "RAG Clinical Chat Assistant":
    st.subheader("Clinical RAG Chatbot & Symptom Checker")
    st.write("Ask inquiries regarding guidelines, symptoms, diets, or drug interaction checks for the 6 diseases.")
    
    # Selected patient context
    patient_context_id = None
    if st.session_state.patients_list:
        pat_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in st.session_state.patients_list}
        selected_pat_str = st.selectbox("Optionally select patient profile context", ["None"] + list(pat_options.keys()))
        if selected_pat_str != "None":
            patient_context_id = pat_options[selected_pat_str]
            
    # Chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    user_message = st.text_input("Enter your clinical question:", key="chat_input_field")
    send_chat = st.button("Query Assistant")
    
    if send_chat and user_message:
        payload = {"message": user_message}
        if patient_context_id:
            payload["patient_id"] = patient_context_id
            
        with st.spinner("Retrieving guidelines and analyzing query..."):
            try:
                r = requests.post(f"{API_URL}/chat", json=payload)
                if r.status_code == 200:
                    ans = r.json()['response']
                    st.session_state.chat_history.append((user_message, ans))
                else:
                    st.error(f"Chat error: {r.text}")
            except Exception as e:
                st.error(f"Failed to query backend: {e}")
                
    # Render chat history
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**👨‍⚕️ Clinical Query:** {q}")
        st.markdown(a)
        st.markdown("---")

# ----------------- TAB 5: BATCH PROCESSING -----------------
elif tab_selection == "Batch Processing":
    st.subheader("Batch Patient Profiles Ingestion")
    st.write("Upload a CSV file containing multiple patient profiles to run predictions in batch.")
    
    csv_template = """name,age,gender,weight_kg,height_cm,glucose,systolic_bp,diastolic_bp,cholesterol,heart_rate,smoking_status,family_history
Alice,45,0,65,160,98,125,82,192,72,0,1
Bob,62,1,85,175,145,142,88,245,80,1,1
Charlie,38,1,72,180,90,118,76,170,68,0,0"""
    
    st.write("#### CSV Format Template")
    st.code(csv_template, language="csv")
    
    uploaded_file = st.file_uploader("Choose patient CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Process Batch Profiles"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            with st.spinner("Processing batch profiles..."):
                try:
                    r = requests.post(f"{API_URL}/batch-predict", files=files)
                    if r.status_code == 200:
                        res = r.json()
                        st.success(f"Batch processed successfully! Total: {res['processed_count']} patient records.")
                        
                        # Display output as dataframe
                        out_df = pd.DataFrame(res['predictions'])
                        st.dataframe(out_df)
                        
                        # Refresh database patient list
                        fetch_patients()
                    else:
                        st.error(f"Batch error: {r.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
