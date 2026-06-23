import os
import json
import yaml
import tempfile
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session as SqlSession
from typing import List

# Local imports
from database.db import get_db, init_db
from database.models import ModelVersion, EvaluationMetric, UserActivityLog
from backend.schemas.applicant import ApplicantInput, PredictionResponse
from backend.services.prediction import PredictionService

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backend_api")

# Initialize database
init_db()

# Initialize FastAPI App
app = FastAPI(
    title="Credit Scoring & Prediction System API",
    description="End-to-end Credit Scoring prediction API with SHAP explainability and model tracking.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction Service Instance
prediction_service = PredictionService()

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_creditworthiness(applicant: ApplicantInput, db: SqlSession = Depends(get_db)):
    """
    Predict Creditworthiness for a single applicant.
    Calculates Credit Score, Risk Category, Approval Recommendation, and probability score.
    Saves applicant record and prediction history to database.
    """
    try:
        applicant_dict = applicant.model_dump()
        result = prediction_service.predict_single(applicant_dict, db)
        
        # Log Activity
        db.add(UserActivityLog(
            action="API Predict Single",
            details=f"Predicted for Applicant ID {result['applicant_id']}. Credit Score: {result['credit_score']}, Risk: {result['risk_category']}"
        ))
        db.commit()
        
        return PredictionResponse(
            credit_score=result["credit_score"],
            risk_category=result["risk_category"],
            approval_recommendation=result["approval_recommendation"],
            probability_score=result["probability_score"],
            applicant_id=result["applicant_id"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Error during prediction endpoint")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/model-metrics", tags=["Metrics"])
def get_model_metrics(db: SqlSession = Depends(get_db)):
    """
    Retrieve all evaluation metrics across model versions.
    """
    try:
        # Load metrics from DB
        versions = db.query(ModelVersion).all()
        result = []
        for ver in versions:
            metrics = db.query(EvaluationMetric).filter(EvaluationMetric.model_version == ver.version).first()
            if metrics:
                result.append({
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
        return result
    except Exception as e:
        logger.exception("Error retrieving model metrics")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance", tags=["Metrics"])
def get_feature_importance():
    """
    Retrieve ranked feature importances for the active model.
    """
    try:
        # Read from metrics.json
        config_path = "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        metrics_path = config['paths']['metrics_path']
        if not os.path.exists(metrics_path):
            raise HTTPException(status_code=404, detail="Metrics/Importance file not found. Ensure model is trained.")
            
        with open(metrics_path, "r") as f:
            data = json.load(f)
            
        importance_dict = data.get("feature_importance", {})
        # Sort feature importances
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": imp} for f, imp in sorted_importance]
    except Exception as e:
        logger.exception("Error retrieving feature importance")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(file: UploadFile = File(...), db: SqlSession = Depends(get_db)):
    """
    Predict creditworthiness for a batch of applicants uploaded via CSV file.
    Returns predicted records.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
        
    try:
        # Save uploaded file to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())
            
        # Run prediction
        df_result = prediction_service.predict_batch_csv(temp_file_path, db)
        
        # Cleanup temporary file
        os.remove(temp_file_path)
        
        # Log activity
        db.add(UserActivityLog(
            action="API Batch Predict",
            details=f"Processed batch CSV {file.filename} with {len(df_result)} rows."
        ))
        db.commit()
        
        # Convert df results to list of dict
        records = df_result.to_dict(orient="records")
        # Handle nan values for JSON conversion
        cleaned_records = []
        for r in records:
            cleaned_row = {}
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    cleaned_row[k] = None
                else:
                    cleaned_row[k] = v
            cleaned_records.append(cleaned_row)
            
        return cleaned_records
    except Exception as e:
        logger.exception("Error processing batch prediction")
        raise HTTPException(status_code=500, detail=f"Failed to process batch: {str(e)}")

# Endpoint to get local SHAP explanation plot for streamlit
@app.get("/predict/{applicant_id}/shap-plot", tags=["Explainability"])
def get_shap_plot(applicant_id: int, db: SqlSession = Depends(get_db)):
    """
    Generate and retrieve the individual SHAP explanation plot for a given prediction history ID.
    """
    # Load config
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Get applicant details and run prediction to draw SHAP
    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")
        
    try:
        # Load prediction_service explainer
        if prediction_service.explainer is None:
            prediction_service.load_artifacts()
            
        if prediction_service.explainer is None:
            raise HTTPException(status_code=400, detail="SHAP Explainer artifact not available")
            
        # Standardize features as dict
        app_dict = {
            col: getattr(applicant, col) 
            for col in (config['features']['numerical'] + config['features']['categorical'])
        }
        
        df_input = pd.DataFrame([app_dict])
        df_preprocessed = prediction_service.preprocessor.transform(df_input)
        df_engineered = add_derived_features(df_preprocessed)
        
        # Drop target if added
        target_col = config['features']['target']
        if target_col in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=[target_col])
            
        # Draw and save local plot
        temp_dir = tempfile.gettempdir()
        plot_path = os.path.join(temp_dir, f"shap_explanation_{applicant_id}.png")
        
        prediction_service.explainer.plot_local_explanation(df_engineered, plot_path)
        
        # Return as FileResponse
        return FileResponse(plot_path, media_type="image/png")
    except Exception as e:
        logger.exception("Error generating local SHAP plot")
        raise HTTPException(status_code=500, detail=str(e))
