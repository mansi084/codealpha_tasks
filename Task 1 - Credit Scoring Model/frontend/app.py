import os
import json
import yaml
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Setup Page Configuration
st.set_page_config(
    page_title="Credit Scoring & Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint definition
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #666;
        margin-top: -15px;
        margin-bottom: 30px;
    }
    
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        padding: 24px;
        margin-bottom: 20px;
        border-left: 5px solid #4e4376;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .score-badge {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 10px 0;
    }
    
    .approve-text {
        color: #2e7d32;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    .review-text {
        color: #f57c00;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    .deny-text {
        color: #d32f2f;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    .risk-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-medium {
        background-color: #fff3e0;
        color: #f57c00;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-high {
        background-color: #ffebee;
        color: #d32f2f;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .progress-bar-container {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        width: 100%;
        margin-top: 10px;
        overflow: hidden;
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check backend API health
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/model-metrics", timeout=2)
        return response.status_code == 200
    except:
        return False

# Load config to get properties
@st.cache_data
def get_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = get_config()

# Navigation Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=80)
st.sidebar.markdown("<h2 style='text-align: center; color: #4e4376;'>Navigation Menu</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.radio(
    "Go To:",
    ["Real-Time Assessment", "Batch Prediction", "Model Performance & Metrics", "Global Feature Importance"]
)

# API Status indicator
api_active = check_api_health()
if api_active:
    st.sidebar.success("Backend API: Connected")
else:
    st.sidebar.warning("Backend API: Not running. (Running local predictions)")

# Main Layout
st.markdown("<div class='main-title'>CREDITWORTHINESS INTELLIGENCE SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Machine Learning & Explainable AI Credit Decisioning Platform</div>", unsafe_allow_html=True)

# ----------------- REAL-TIME ASSESSMENT TAB -----------------
if app_mode == "Real-Time Assessment":
    st.markdown("### Applicant Risk Assessment Form")
    st.write("Fill in the financial attributes below to evaluate the creditworthiness of an applicant in real-time.")
    
    # Form layout inside columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Personal Details & Basic Info")
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        employment_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, max_value=1000000.0, value=65000.0, step=1000.0)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, max_value=100000.0, value=int(annual_income/12), step=100.0)
        savings_balance = st.number_input("Savings Account Balance ($)", min_value=0.0, max_value=2000000.0, value=8500.0, step=500.0)

    with col2:
        st.markdown("##### Liabilities & Desired Loan")
        loan_amount = st.number_input("Desired Loan Amount ($)", min_value=0.0, max_value=1000000.0, value=25000.0, step=1000.0)
        existing_debts = st.number_input("Existing Debts ($)", min_value=0.0, max_value=1000000.0, value=12000.0, step=500.0)
        # Calculate auto debt to income ratio
        debt_to_income_ratio = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=10.0, value=round(existing_debts / max(1.0, annual_income), 4), step=0.01)

    with col3:
        st.markdown("##### Credit Behavior & History")
        number_of_credit_cards = st.number_input("Number of Credit Cards", min_value=0, max_value=20, value=3, step=1)
        credit_utilization_ratio = st.slider("Credit Card Utilization Ratio", min_value=0.0, max_value=1.5, value=0.35, step=0.01)
        credit_history_length = st.number_input("Credit History Length (Years)", min_value=0.0, max_value=60.0, value=10.0, step=0.5)
        payment_history = st.selectbox("Payment History Category", ["Excellent", "Good", "Fair", "Poor"], index=1)
        number_of_late_payments = st.number_input("Number of Late Payments", min_value=0, max_value=100, value=1, step=1)
        loan_repayment_history = st.selectbox("Previous Loan Repayments", ["All Paid", "Mostly Paid", "Delayed", "Defaulted"], index=1)
        previous_defaults = st.selectbox("Has Previous Defaults?", ["No", "Yes"], index=0)

    # Submission Action
    if st.button("Evaluate Credit Risk", type="primary", use_container_width=True):
        payload = {
            "annual_income": float(annual_income),
            "monthly_income": float(monthly_income),
            "loan_amount": float(loan_amount),
            "existing_debts": float(existing_debts),
            "debt_to_income_ratio": float(debt_to_income_ratio),
            "number_of_credit_cards": int(number_of_credit_cards),
            "credit_utilization_ratio": float(credit_utilization_ratio),
            "payment_history": payment_history,
            "number_of_late_payments": int(number_of_late_payments),
            "loan_repayment_history": loan_repayment_history,
            "employment_length": float(employment_length),
            "age": int(age),
            "savings_balance": float(savings_balance),
            "previous_defaults": previous_defaults,
            "credit_history_length": float(credit_history_length)
        }
        
        with st.spinner("Processing applicant profile & running machine learning analysis..."):
            # Call API or local service
            result = None
            if api_active:
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.warning(f"Failed to communicate with API: {e}. Falling back to local.")
                    
            if result is None:
                # Local execution fallback
                from database.db import Session
                from backend.services.prediction import PredictionService
                try:
                    local_svc = PredictionService()
                    db = Session()
                    result = local_svc.predict_single(payload, db)
                    db.close()
                except Exception as ex:
                    st.error(f"Failed local prediction: {ex}")
            
            if result:
                # Show results container
                st.markdown("---")
                st.markdown("### Creditworthiness Assessment Report")
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    score = result["credit_score"]
                    risk = result["risk_category"]
                    rec = result["approval_recommendation"]
                    prob = result["probability_score"]
                    app_id = result["applicant_id"]
                    
                    # Color formatting
                    if rec == "Approved":
                        rec_class = "approve-text"
                    elif rec == "Review":
                        rec_class = "review-text"
                    else:
                        rec_class = "deny-text"
                        
                    if risk == "Low Risk":
                        risk_class = "risk-low"
                        prog_color = "#2e7d32"
                    elif risk == "Medium Risk":
                        risk_class = "risk-medium"
                        prog_color = "#f57c00"
                    else:
                        risk_class = "risk-high"
                        prog_color = "#d32f2f"
                        
                    percentage_score = (score - 300) / 550 * 100
                    
                    st.markdown(f"""
                    <div class="card">
                        <h4>Calculated Credit Score</h4>
                        <div class="score-badge" style="color: {prog_color};">{score}</div>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: {percentage_score}%; background-color: {prog_color};"></div>
                        </div>
                        <p style="text-align: center; color: #888; font-size: 0.8rem; margin-top: 5px;">Score Scale: 300 to 850</p>
                    </div>
                    
                    <div class="card">
                        <h4>Risk Profile & Recommendation</h4>
                        <p style="margin-top: 10px;">Risk Profile: <span class="{risk_class}">{risk}</span></p>
                        <p style="margin-top: 10px;">Approval Status: <span class="{rec_class}">{rec}</span></p>
                        <p style="margin-top: 10px; font-size: 0.9rem; color: #666;">Probability Score: <b>{prob:.2%}</b></p>
                        <p style="font-size: 0.8rem; color: #aaa; margin-top: 15px;">Applicant Reference ID: #{app_id}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown("#### Explainable AI Prediction Breakdown")
                    st.write("Below is the local SHAP explanation plot. It shows which factors contributed to this applicant's final credit score rating.")
                    
                    # Fetch and show SHAP plot from API or create locally
                    shap_loaded = False
                    if api_active:
                        try:
                            plot_url = f"{API_URL}/predict/{app_id}/shap-plot"
                            st.image(plot_url, caption="Local SHAP Explanation (Factors boosting score points show positive contributions, dragging factors show negative)", use_column_width=True)
                            shap_loaded = True
                        except Exception as e:
                            logger.error(f"Error fetching SHAP image from API: {e}")
                            
                    if not shap_loaded:
                        # Draw locally
                        try:
                            from backend.services.prediction import PredictionService
                            from database.db import Session
                            local_svc = PredictionService()
                            
                            if local_svc.explainer:
                                df_input = pd.DataFrame([payload])
                                df_prepped = local_svc.preprocessor.transform(df_input)
                                df_eng = add_derived_features(df_prepped)
                                target_col = config['features']['target']
                                if target_col in df_eng.columns:
                                    df_eng = df_eng.drop(columns=[target_col])
                                    
                                import tempfile
                                temp_path = os.path.join(tempfile.gettempdir(), f"local_shap_{app_id}.png")
                                local_svc.explainer.plot_local_explanation(df_eng, temp_path)
                                st.image(temp_path, caption="Local SHAP Explanation Plot", use_column_width=True)
                            else:
                                st.info("SHAP explainer model is not available to render explanation plot.")
                        except Exception as ex:
                            st.error(f"Could not render explanation plot: {ex}")

# ----------------- BATCH PREDICTION TAB -----------------
elif app_mode == "Batch Prediction":
    st.markdown("### Upload Batch Applicant File")
    st.write("Upload a CSV file containing applicant profiles. The system will predict creditworthiness for all applicants, save records to the history database, and return the annotated files.")
    
    # Download sample template
    st.markdown("##### Download Sample Data Template")
    sample_data = pd.DataFrame([{
        "annual_income": 80000.0,
        "monthly_income": 6666.0,
        "loan_amount": 35000.0,
        "existing_debts": 15000.0,
        "debt_to_income_ratio": 0.1875,
        "number_of_credit_cards": 4,
        "credit_utilization_ratio": 0.3,
        "payment_history": "Excellent",
        "number_of_late_payments": 0,
        "loan_repayment_history": "All Paid",
        "employment_length": 8.0,
        "age": 42,
        "savings_balance": 25000.0,
        "previous_defaults": "No",
        "credit_history_length": 15.0
    }, {
        "annual_income": 45000.0,
        "monthly_income": 3750.0,
        "loan_amount": 60000.0,
        "existing_debts": 35000.0,
        "debt_to_income_ratio": 0.77,
        "number_of_credit_cards": 7,
        "credit_utilization_ratio": 0.88,
        "payment_history": "Poor",
        "number_of_late_payments": 6,
        "loan_repayment_history": "Delayed",
        "employment_length": 1.5,
        "age": 27,
        "savings_balance": 400.0,
        "previous_defaults": "Yes",
        "credit_history_length": 3.2
    }])
    
    st.download_button(
        label="Download Template CSV",
        data=sample_data.to_csv(index=False),
        file_name="credit_applicant_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Process Batch Predictions", type="primary"):
            with st.spinner("Processing batch file..."):
                result_df = None
                
                # Check backend API first
                if api_active:
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        response = requests.post(f"{API_URL}/batch-predict", files=files)
                        if response.status_code == 200:
                            result_df = pd.DataFrame(response.json())
                        else:
                            st.error(f"API Error: {response.text}")
                    except Exception as e:
                        st.warning(f"API Batch Request failed: {e}. Running locally.")
                        
                if result_df is None:
                    # Fallback to local prediction
                    try:
                        import tempfile
                        from database.db import Session
                        from backend.services.prediction import PredictionService
                        
                        temp_dir = tempfile.gettempdir()
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                            
                        local_svc = PredictionService()
                        db = Session()
                        result_df = local_svc.predict_batch_csv(temp_path, db)
                        db.close()
                        os.remove(temp_path)
                    except Exception as ex:
                        st.error(f"Local batch prediction failed: {ex}")
                        
                if result_df is not None:
                    st.success("Batch completed successfully!")
                    
                    # Highlight critical results
                    total_count = len(result_df)
                    approved_count = len(result_df[result_df['approval_recommendation'] == 'Approved'])
                    review_count = len(result_df[result_df['approval_recommendation'] == 'Review'])
                    denied_count = len(result_df[result_df['approval_recommendation'] == 'Denied'])
                    
                    st.markdown("#### Batch Summary Metrics")
                    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                    b_col1.metric("Total Applicants", total_count)
                    b_col2.metric("Approved", f"{approved_count} ({approved_count/total_count:.1%})")
                    b_col3.metric("Need Manual Review", f"{review_count} ({review_count/total_count:.1%})")
                    b_col4.metric("Denied", f"{denied_count} ({denied_count/total_count:.1%})")
                    
                    st.markdown("#### Results Dataset")
                    st.dataframe(result_df)
                    
                    st.download_button(
                        label="Download Annotated Results CSV",
                        data=result_df.to_csv(index=False),
                        file_name="credit_predictions_output.csv",
                        mime="text/csv"
                    )

# ----------------- MODEL PERFORMANCE & METRICS TAB -----------------
elif app_mode == "Model Performance & Metrics":
    st.markdown("### Model Performance Dashboard")
    st.write("View the evaluation performance metrics of active and historical models registered in the model store.")
    
    # Retrieve metrics from API/local database
    metrics_list = []
    if api_active:
        try:
            response = requests.get(f"{API_URL}/model-metrics")
            if response.status_code == 200:
                metrics_list = response.json()
        except:
            pass
            
    if not metrics_list:
        # Load locally
        from database.db import Session
        from database.models import ModelVersion, EvaluationMetric
        db = Session()
        try:
            versions = db.query(ModelVersion).all()
            for ver in versions:
                metrics = db.query(EvaluationMetric).filter(EvaluationMetric.model_version == ver.version).first()
                if metrics:
                    metrics_list.append({
                        "version": ver.version,
                        "model_name": ver.model_name,
                        "trained_at": ver.trained_at.isoformat(),
                        "is_active": ver.is_active,
                        "metrics": {
                            "accuracy": metrics.accuracy,
                            "precision": metrics.precision,
                            "recall": metrics.recall,
                            "f1_score": metrics.f1_score,
                            "roc_auc": metrics.roc_auc
                        }
                    })
        except Exception as e:
            st.error(f"Error reading local database: {e}")
        finally:
            db.close()
            
    if metrics_list:
        # Create metrics table
        records = []
        for item in metrics_list:
            records.append({
                "Version": item["version"],
                "Model Algorithm": item["model_name"],
                "Trained Date": item["trained_at"][:10],
                "Is Active": "✅ Yes" if item["is_active"] else "No",
                "Accuracy": f"{item['metrics']['accuracy']:.4f}",
                "Precision": f"{item['metrics']['precision']:.4f}",
                "Recall": f"{item['metrics']['recall']:.4f}",
                "F1 Score": f"{item['metrics']['f1_score']:.4f}",
                "ROC-AUC": f"{item['metrics']['roc_auc']:.4f}"
            })
        st.markdown("#### Registered Models Comparison")
        st.table(pd.DataFrame(records))
        
        # Display active curves
        st.markdown("#### Performance Evaluation Curves (Active Model)")
        st.write("ROC Curve and Precision-Recall Curves evaluate classification performance at different thresholds.")
        
        p_col1, p_col2 = st.columns(2)
        
        plots_dir = config['paths']['plots_dir']
        roc_path = os.path.join(plots_dir, "roc_curve.png")
        pr_path = os.path.join(plots_dir, "precision_recall_curve.png")
        cm_path = os.path.join(plots_dir, "confusion_matrix.png")
        
        with p_col1:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Curve", use_column_width=True)
            else:
                st.info("ROC Curve image not available yet. Complete training process.")
                
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix Heatmap", use_column_width=True)
                
        with p_col2:
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Precision-Recall Curve", use_column_width=True)
            else:
                st.info("Precision-Recall Curve image not available yet.")
    else:
        st.info("No trained models found in the database. Run training pipeline to populate metrics.")

# ----------------- GLOBAL FEATURE IMPORTANCE TAB -----------------
elif app_mode == "Global Feature Importance":
    st.markdown("### Global Feature Importance Analysis")
    st.write("Understand which features are overall most important in predicting applicant creditworthiness across the active model.")
    
    plots_dir = config['paths']['plots_dir']
    fi_path = os.path.join(plots_dir, "feature_importance.png")
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("#### Feature Importance Heatmap")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Ranked Feature Importance (Active Model)", use_column_width=True)
        else:
            st.info("Feature importance plot not found. Ensure model is trained.")
            
    with col_r:
        st.markdown("#### Tabular Feature Importance")
        importance_list = []
        if api_active:
            try:
                response = requests.get(f"{API_URL}/feature-importance")
                if response.status_code == 200:
                    importance_list = response.json()
            except:
                pass
                
        if not importance_list:
            # Load local metrics.json
            metrics_path = config['paths']['metrics_path']
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                    importance_dict = data.get("feature_importance", {})
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    importance_list = [{"feature": f, "importance": imp} for f, imp in sorted_importance]
                except Exception as e:
                    st.error(f"Error loading local metrics: {e}")
                    
        if importance_list:
            fi_df = pd.DataFrame(importance_list)
            fi_df.columns = ["Feature Name", "Relative Importance Score"]
            # format values
            fi_df["Relative Importance Score"] = fi_df["Relative Importance Score"].map(lambda x: f"{x:.4%}" if x <= 1.0 else f"{x:.4f}")
            st.dataframe(fi_df, use_container_width=True)
        else:
            st.info("No feature importance metrics available yet.")
