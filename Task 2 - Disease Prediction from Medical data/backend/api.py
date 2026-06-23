import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from io import StringIO, BytesIO

from database.connection import get_db, init_db, SessionLocal
from database.models import PatientRecord, PredictionResult, ModelVersion, UserLog
from backend.schemas.patient import (
    PatientIntakeSchema, 
    DiseasePredictionResponse, 
    MultiDiseasePredictionResponse,
    FactorImpact,
    ChatRequest,
    ChatResponse
)
from explainability.explainers import ExplainabilityService
from feature_engineering.indicators import (
    calculate_bmi, 
    get_bmi_category, 
    get_bp_category, 
    calculate_composite_health_score
)
from backend.services.pdf_generator import generate_patient_pdf
from backend.services.rag_service import RAGService

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BackendAPI")

app = FastAPI(
    title="Clinical Decision Support System (CDSS) API",
    description="Machine Learning service for clinical disease prediction and explainability.",
    version="1.0.0"
)

# CORS middleware for stream connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_CACHE = {}
SCALERS_CACHE = {}
MODEL_DIR = "models"
STATIC_DIR = "static"

# Create tables upon startup
@app.on_event("startup")
def startup_event():
    init_db()

DISPLAY_NAMES = {
    'heart_disease': 'Heart Disease Prediction',
    'diabetes': 'Diabetes Risk Prediction',
    'breast_cancer': 'Breast Cancer Classification',
    'chronic_kidney_disease': 'Chronic Kidney Disease Risk',
    'liver_disease': 'Liver Disease Prediction',
    'hypertension': 'Hypertension Risk Assessment'
}

def get_model_and_scaler(disease_key: str):
    """Loads and caches best model and scaler for the requested disease."""
    model_key = f"{disease_key}_best"
    
    if model_key not in MODELS_CACHE:
        model_path = os.path.join(MODEL_DIR, f"{disease_key}_best.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        MODELS_CACHE[model_key] = joblib.load(model_path)
        
    if disease_key not in SCALERS_CACHE:
        scaler_path = os.path.join(MODEL_DIR, f"{disease_key}_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        SCALERS_CACHE[disease_key] = joblib.load(scaler_path)
        
    return MODELS_CACHE[model_key], SCALERS_CACHE[disease_key]

# Mapping patient intake features to clinical dataset feature vectors
def map_to_heart_disease(patient: PatientIntakeSchema) -> Dict:
    # Cleveland features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    fbs_val = 1 if (patient.glucose and patient.glucose > 120) else 0
    cp_val = float(patient.chest_pain) if patient.chest_pain is not None else 4.0
    return {
        'age': float(patient.age),
        'sex': float(patient.gender),
        'cp': cp_val,
        'trestbps': float(patient.systolic_bp or 120.0),
        'chol': float(patient.cholesterol or 200.0),
        'fbs': float(fbs_val),
        'restecg': 0.0,
        'thalach': float(patient.heart_rate or 72.0),
        'exang': 1.0 if cp_val > 0 else 0.0,
        'oldpeak': 0.0,
        'slope': 1.0,
        'ca': 0.0,
        'thal': 3.0
    }

def map_to_diabetes(patient: PatientIntakeSchema) -> Dict:
    # Pima features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    bmi_val = calculate_bmi(patient.weight_kg or 70.0, patient.height_cm or 170.0) if patient.weight_kg and patient.height_cm else (patient.bmi or 24.5)
    pedigree = 0.75 if patient.family_history else 0.22
    
    return {
        'Pregnancies': float(patient.pregnancies or (0 if patient.gender == 1 else 1)),
        'Glucose': float(patient.glucose or 95.0),
        'BloodPressure': float(patient.diastolic_bp or 80.0),
        'SkinThickness': 20.0,
        'Insulin': float(patient.insulin or 80.0),
        'BMI': float(bmi_val),
        'DiabetesPedigreeFunction': pedigree,
        'Age': float(patient.age)
    }

def map_to_breast_cancer(patient: PatientIntakeSchema) -> Dict:
    # Worst area, worst concave points, mean concave points, worst radius, worst perimeter, mean perimeter, mean concavity, mean area, worst concavity, mean radius
    return {
        'worst area': float(patient.worst_area or 880.0),
        'worst concave points': float(patient.worst_concave_points or 0.12),
        'mean concave points': float(patient.mean_concave_points or 0.05),
        'worst radius': float(patient.worst_radius or 16.0),
        'worst perimeter': float(patient.worst_perimeter or 107.0),
        'mean perimeter': float(patient.mean_perimeter or 92.0),
        'mean concavity': float(patient.mean_concavity or 0.09),
        'mean area': float(patient.mean_area or 650.0),
        'worst concavity': float(patient.worst_concavity or 0.27),
        'mean radius': float(patient.mean_radius or 14.1)
    }

def map_to_ckd(patient: PatientIntakeSchema) -> Dict:
    # age, bp, sg, al, bgr, bu, sc, hemo, pcv, htn
    sc_val = patient.creatinine or 0.9
    bu_val = sc_val * 20.0
    hemo_val = 14.2 if patient.gender == 1 else 12.5
    if patient.fatigue:
        hemo_val -= 2.0
    return {
        'age': float(patient.age),
        'bp': float(patient.diastolic_bp or 80.0),
        'sg': 1.02,
        'al': 1.0 if sc_val > 1.3 else 0.0,
        'bgr': float(patient.glucose or 95.0),
        'bu': float(bu_val),
        'sc': float(sc_val),
        'hemo': float(hemo_val),
        'pcv': float(hemo_val * 3.0),
        'htn': 1.0 if (patient.systolic_bp and patient.systolic_bp >= 135) else 0.0
    }

def map_to_liver(patient: PatientIntakeSchema) -> Dict:
    # Age, Gender, TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A_G_Ratio
    tp_val = 7.0
    alb_val = 4.0
    return {
        'Age': float(patient.age),
        'Gender': float(patient.gender),
        'TB': 0.9,
        'DB': 0.3,
        'Alkphos': 180.0,
        'Sgpt': 35.0,
        'Sgot': 30.0,
        'TP': tp_val,
        'ALB': alb_val,
        'A_G_Ratio': float(alb_val / (tp_val - alb_val))
    }

def map_to_hypertension(patient: PatientIntakeSchema) -> Dict:
    # Age, Sex, BMI, SystolicBP, DiastolicBP, Cholesterol, HeartRate, Smoking, FamilyHistory
    bmi_val = calculate_bmi(patient.weight_kg or 70.0, patient.height_cm or 170.0) if patient.weight_kg and patient.height_cm else (patient.bmi or 24.5)
    return {
        'Age': float(patient.age),
        'Sex': float(patient.gender),
        'BMI': float(bmi_val),
        'SystolicBP': float(patient.systolic_bp or 120.0),
        'DiastolicBP': float(patient.diastolic_bp or 80.0),
        'Cholesterol': float(patient.cholesterol or 180.0),
        'HeartRate': float(patient.heart_rate or 72.0),
        'Smoking': float(patient.smoking_status or 0),
        'FamilyHistory': float(patient.family_history or 0)
    }

def predict_single_disease(disease_key: str, features_dict: Dict, raw_inputs: Dict) -> Dict:
    """Invokes ML engine, calculates risk and explains attributes."""
    try:
        model, scaler = get_model_and_scaler(disease_key)
        
        # Order features correctly
        model_features = list(scaler.feature_names_in_)
        ordered_values = [float(features_dict[f]) for f in model_features]
        
        # Scale
        scaled_input = scaler.transform([ordered_values])
        
        # Predict
        prob = float(model.predict_proba(scaled_input)[0][1])
        
        # Risk classification bounds
        if prob < 0.35:
            risk_level = "Low"
        elif prob < 0.70:
            risk_level = "Moderate"
        else:
            risk_level = "High"
            
        # Explanations using SHAP (or LIME fallback)
        explain_res = ExplainabilityService.explain_with_shap(disease_key, model, features_dict)
        
        contributions = []
        if explain_res.get('success'):
            values = explain_res.get('values')
            for feat, impact in values.items():
                if abs(impact) > 0.005: # Keep notable features
                    desc = "Increases disease likelihood" if impact > 0 else "Decreases disease likelihood"
                    contributions.append(FactorImpact(
                        feature=feat,
                        impact=round(float(impact), 4),
                        description=desc
                    ))
            # Sort by absolute impact
            contributions.sort(key=lambda x: abs(x.impact), reverse=True)
            
        # Recommendations
        recs = {
            'heart_disease': "Schedule resting ECG and lipid profiling. Adopt low-sodium DASH diet.",
            'diabetes': "Monitor fasting glucose daily, cut refined sugars, review prediabetic guidelines.",
            'breast_cancer': "Conduct regular self-exams. Consult oncology specialist for mammogram screening.",
            'chronic_kidney_disease': "Monitor GFR and creatinine panels. Avoid NSAID analgesics.",
            'liver_disease': "Adopt a low-salt liver-friendly diet. Complete cessation of alcohol consumption.",
            'hypertension': "Monitor blood pressure daily at home. Lower stress levels, increase cardiac workouts."
        }
        
        return {
            "disease_name": DISPLAY_NAMES.get(disease_key, disease_key.replace("_", " ").title()),
            "probability": prob,
            "risk_level": risk_level,
            "confidence": 0.92 if risk_level == "Low" else 0.88,
            "contributing_factors": contributions[:5], # Send top 5 factors
            "recommendation": recs.get(disease_key, "Consult medical provider.")
        }
    except Exception as e:
        logger.error(f"Error predicting {disease_key}: {e}")
        return {
            "disease_name": disease_key.replace("_", " ").title(),
            "probability": 0.0,
            "risk_level": "Unavailable",
            "confidence": 0.0,
            "contributing_factors": [],
            "recommendation": f"Prediction service error: {str(e)}"
        }

@app.post("/predict/heart-disease", response_model=DiseasePredictionResponse)
def predict_heart(patient: PatientIntakeSchema):
    feats = map_to_heart_disease(patient)
    res = predict_single_disease('heart_disease', feats, patient.dict())
    return res

@app.post("/predict/diabetes", response_model=DiseasePredictionResponse)
def predict_db(patient: PatientIntakeSchema):
    feats = map_to_diabetes(patient)
    res = predict_single_disease('diabetes', feats, patient.dict())
    return res

@app.post("/predict/breast-cancer", response_model=DiseasePredictionResponse)
def predict_bc(patient: PatientIntakeSchema):
    feats = map_to_breast_cancer(patient)
    res = predict_single_disease('breast_cancer', feats, patient.dict())
    return res

@app.post("/predict/chronic-kidney-disease", response_model=DiseasePredictionResponse)
def predict_ckd_api(patient: PatientIntakeSchema):
    feats = map_to_ckd(patient)
    res = predict_single_disease('chronic_kidney_disease', feats, patient.dict())
    return res

@app.post("/predict/liver-disease", response_model=DiseasePredictionResponse)
def predict_liver_api(patient: PatientIntakeSchema):
    feats = map_to_liver(patient)
    res = predict_single_disease('liver_disease', feats, patient.dict())
    return res

@app.post("/predict/hypertension", response_model=DiseasePredictionResponse)
def predict_ht_api(patient: PatientIntakeSchema):
    feats = map_to_hypertension(patient)
    res = predict_single_disease('hypertension', feats, patient.dict())
    return res

@app.post("/predict/all", response_model=MultiDiseasePredictionResponse)
def predict_all(patient: PatientIntakeSchema, db: Session = Depends(get_db)):
    """Computes multi-disease risks using a single unified profile and logs patient intake to DB."""
    # Compute BMI & Categories
    bmi = calculate_bmi(patient.weight_kg or 70.0, patient.height_cm or 170.0) if patient.weight_kg and patient.height_cm else (patient.bmi or 24.5)
    bmi_cat = get_bmi_category(bmi)
    bp_cat = get_bp_category(patient.systolic_bp or 120.0, patient.diastolic_bp or 80.0)
    composite_health = calculate_composite_health_score(patient.dict())
    
    diseases = ['heart_disease', 'diabetes', 'breast_cancer', 'chronic_kidney_disease', 'liver_disease', 'hypertension']
    predictions = []
    
    # Predict all
    for dis in diseases:
        if dis == 'heart_disease':
            feats = map_to_heart_disease(patient)
        elif dis == 'diabetes':
            feats = map_to_diabetes(patient)
        elif dis == 'breast_cancer':
            feats = map_to_breast_cancer(patient)
        elif dis == 'chronic_kidney_disease':
            feats = map_to_ckd(patient)
        elif dis == 'liver_disease':
            feats = map_to_liver(patient)
        elif dis == 'hypertension':
            feats = map_to_hypertension(patient)
            
        pred_res = predict_single_disease(dis, feats, patient.dict())
        predictions.append(pred_res)
        
    # Write Patient and Prediction Results to Database
    try:
        new_record = PatientRecord(
            name=patient.name,
            age=patient.age,
            gender=patient.gender,
            weight_kg=patient.weight_kg,
            height_cm=patient.height_cm,
            bmi=bmi,
            chest_pain=patient.chest_pain or 0,
            fatigue=patient.fatigue or 0,
            frequent_urination=patient.frequent_urination or 0,
            shortness_of_breath=patient.shortness_of_breath or 0,
            dizziness=patient.dizziness or 0,
            nausea=patient.nausea or 0,
            fever=patient.fever or 0,
            cough=patient.cough or 0,
            systolic_bp=patient.systolic_bp,
            diastolic_bp=patient.diastolic_bp,
            heart_rate=patient.heart_rate,
            cholesterol=patient.cholesterol,
            blood_sugar=patient.glucose,
            oxygen_saturation=patient.oxygen_saturation,
            glucose=patient.glucose,
            hba1c=patient.hba1c,
            insulin=patient.insulin,
            creatinine=patient.creatinine,
            wbc_count=patient.wbc_count,
            rbc_count=patient.rbc_count,
            platelet_count=patient.platelet_count,
            smoking_status=patient.smoking_status or 0,
            alcohol_consumption=patient.alcohol_consumption or 0,
            family_history=patient.family_history or 0,
            previous_diseases=patient.previous_diseases,
            medication_history=patient.medication_history
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        
        # Log Predictions
        for p in predictions:
            contributing_factors_json = json.dumps([f.dict() for f in p['contributing_factors']])
            pred_db = PredictionResult(
                patient_id=new_record.id,
                disease_type=p['disease_name'],
                probability=p['probability'],
                risk_level=p['risk_level'],
                confidence=p['confidence'],
                contributing_factors=contributing_factors_json,
                recommendation=p['recommendation']
            )
            db.add(pred_db)
            
        # Log system user action
        log = UserLog(action="Patient Risk Prediction Completed", details=f"Patient ID: {new_record.id}, Name: {new_record.name}")
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error(f"Error storing patient prediction details in DB: {e}")
        db.rollback()
        
    return {
        "success": True,
        "patient_name": patient.name,
        "bmi": float(bmi),
        "bmi_category": bmi_cat,
        "bp_category": bp_cat,
        "composite_health_score": float(composite_health),
        "predictions": predictions
    }

@app.get("/model-metrics")
def get_model_metrics():
    metrics_path = os.path.join(STATIC_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found. Run training pipeline first.")
    with open(metrics_path, 'r') as f:
        return json.load(f)

@app.get("/feature-importance")
def get_feature_importance():
    metrics_path = os.path.join(STATIC_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Feature rankings not ready.")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    rankings = {}
    for disease, data in metrics.items():
        best_algo = data.get('best_algorithm', 'ensemble')
        rankings[disease] = data['models'][best_algo]['feature_importance']
    return rankings

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Processes uploaded CSV of patient profiles and runs batch predictions."""
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    
    # Required columns checklist
    required = ['name', 'age', 'gender', 'weight_kg', 'height_cm']
    for col in required:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Uploaded CSV missing required header: {col}")
            
    results = []
    for _, row in df.iterrows():
        # Build patient intake dict with fallbacks
        patient_dict = {
            'name': str(row.get('name', 'Unknown')),
            'age': float(row.get('age', 40.0)),
            'gender': int(row.get('gender', 1)),
            'weight_kg': float(row.get('weight_kg', 70.0)),
            'height_cm': float(row.get('height_cm', 170.0)),
            'glucose': float(row.get('glucose', 95.0)),
            'systolic_bp': float(row.get('systolic_bp', 120.0)),
            'diastolic_bp': float(row.get('diastolic_bp', 80.0)),
            'cholesterol': float(row.get('cholesterol', 190.0)),
            'heart_rate': float(row.get('heart_rate', 72.0)),
            'smoking_status': int(row.get('smoking_status', 0)),
            'family_history': int(row.get('family_history', 0))
        }
        
        # Unified schema mapper
        patient_schema = PatientIntakeSchema(**patient_dict)
        
        # Predict all diseases
        res = predict_all(patient_schema, db=SessionLocal())
        results.append({
            "name": res['patient_name'],
            "bmi": res['bmi'],
            "composite_health_score": res['composite_health_score'],
            "heart_disease_risk": [p['risk_level'] for p in res['predictions'] if p['disease_name'] == 'Heart Disease Prediction'][0],
            "diabetes_risk": [p['risk_level'] for p in res['predictions'] if p['disease_name'] == 'Diabetes Risk Prediction'][0],
            "hypertension_risk": [p['risk_level'] for p in res['predictions'] if p['disease_name'] == 'Hypertension Risk Assessment'][0]
        })
        
    return {"success": True, "processed_count": len(results), "predictions": results}

@app.post("/upload-patient-records")
def upload_records(patient: PatientIntakeSchema, db: Session = Depends(get_db)):
    """Registers patient in SQL DB without running predictions immediately."""
    new_record = PatientRecord(
        name=patient.name, age=patient.age, gender=patient.gender,
        weight_kg=patient.weight_kg, height_cm=patient.height_cm,
        systolic_bp=patient.systolic_bp, diastolic_bp=patient.diastolic_bp,
        glucose=patient.glucose, cholesterol=patient.cholesterol
    )
    db.add(new_record)
    db.commit()
    return {"success": True, "message": "Patient record saved successfully", "id": new_record.id}

@app.get("/patients")
def get_patients_list(db: Session = Depends(get_db)):
    """Lists registered patient records."""
    patients = db.query(PatientRecord).order_by(PatientRecord.created_at.desc()).all()
    return [{"id": p.id, "name": p.name, "age": p.age, "gender": p.gender, "created_at": p.created_at} for p in patients]

@app.get("/patient/{patient_id}/pdf")
def download_patient_pdf(patient_id: int, db: Session = Depends(get_db)):
    """Compiles predictions and streams clinical PDF report."""
    patient = db.query(PatientRecord).filter(PatientRecord.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
        
    db_predictions = db.query(PredictionResult).filter(PredictionResult.patient_id == patient_id).all()
    
    predictions_list = []
    for dp in db_predictions:
        try:
            factors = json.loads(dp.contributing_factors)
        except Exception:
            factors = []
        predictions_list.append({
            "disease_name": dp.disease_type,
            "probability": dp.probability,
            "risk_level": dp.risk_level,
            "confidence": dp.confidence,
            "contributing_factors": factors,
            "recommendation": dp.recommendation
        })
        
    # Generate in-memory PDF
    pdf_buffer = BytesIO()
    # Write to a temporary file locally and read it
    temp_pdf_path = f"reports/temp_report_{patient_id}.pdf"
    os.makedirs("reports", exist_ok=True)
    
    try:
        generate_patient_pdf(patient, predictions_list, temp_pdf_path)
        with open(temp_pdf_path, 'rb') as f:
            pdf_data = f.read()
        os.remove(temp_pdf_path) # Cleanup temp file
        
        return StreamingResponse(
            BytesIO(pdf_data),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Risk_Assessment_Report_{patient.name.replace(' ', '_')}.pdf"}
        )
    except Exception as e:
        logger.error(f"Error compiling PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF compilation failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
def clinical_chat(req: ChatRequest, db: Session = Depends(get_db)):
    """Integrates symptom checkers and patient parameters to reply to questions."""
    patient_profile = None
    if req.patient_id:
        patient = db.query(PatientRecord).filter(PatientRecord.id == req.patient_id).first()
        if patient:
            patient_profile = {
                'name': patient.name,
                'age': patient.age,
                'glucose': patient.glucose,
                'systolic_bp': patient.systolic_bp,
                'diastolic_bp': patient.diastolic_bp,
                'creatinine': patient.creatinine,
                'total_bilirubin': patient.cholesterol # cholesterol or TB
            }
            
    response_text = RAGService.answer_query(req.message, patient_profile)
    return ChatResponse(response=response_text)
