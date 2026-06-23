import os
import yaml
import logging
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session as SqlSession

# Local modules
from preprocessing.preprocessor import Preprocessor
from feature_engineering.engineer import add_derived_features
from explainability.explainer import CreditExplainer
from database.models import Applicant, PredictionHistory, ModelVersion

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        # Load Config
        config_path = "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.best_model_path = self.config['paths']['best_model_path']
        self.preprocessor_path = self.config['paths']['preprocessor_path']
        self.explainer_path = self.config['paths']['explainer_path']
        
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.model_version_str = "unknown"
        
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained model, preprocessor, and explainer from disk."""
        logger.info("Loading prediction service artifacts...")
        if os.path.exists(self.preprocessor_path):
            self.preprocessor = Preprocessor.load(self.preprocessor_path)
        else:
            logger.warning("Preprocessor artifact not found.")

        if os.path.exists(self.best_model_path):
            import pickle
            with open(self.best_model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            logger.warning("Best model artifact not found.")

        if os.path.exists(self.explainer_path):
            self.explainer = CreditExplainer.load(self.explainer_path)
        else:
            logger.warning("SHAP explainer artifact not found.")
            
        # Get active model version from DB
        from database.db import Session
        db = Session()
        try:
            active_ver = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
            if active_ver:
                self.model_version_str = active_ver.version
                logger.info(f"Active model version: {self.model_version_str}")
        except Exception as e:
            logger.error(f"Error querying active version: {e}")
        finally:
            db.close()

    def calculate_credit_score(self, prob_good: float) -> int:
        """Map probability score [0, 1] to FICO-like scale 300 to 850."""
        score = 300 + int(550 * prob_good)
        return int(np.clip(score, 300, 850))

    def determine_risk_category(self, credit_score: int) -> str:
        """Determine risk level based on Credit Score."""
        if credit_score >= 700:
            return "Low Risk"
        elif credit_score >= 580:
            return "Medium Risk"
        else:
            return "High Risk"

    def determine_approval_recommendation(self, credit_score: int) -> str:
        """Determine recommendation based on Credit Score."""
        if credit_score >= 680:
            return "Approved"
        elif credit_score >= 580:
            return "Review"
        else:
            return "Denied"

    def predict_single(self, applicant_data: dict, db: SqlSession) -> dict:
        """
        Process applicant data, store in DB, predict credit score/risk, and record history.
        """
        if self.model is None or self.preprocessor is None:
            # Try reloading in case they were generated since initialization
            self.load_artifacts()
            if self.model is None or self.preprocessor is None:
                raise ValueError("Model artifacts are not loaded. Please run training pipeline first.")

        # 1. Save applicant raw data to Database
        db_applicant = Applicant(**applicant_data)
        db.add(db_applicant)
        db.commit()
        db.refresh(db_applicant)
        
        # 2. Preprocess & Feature Engineer
        df_input = pd.DataFrame([applicant_data])
        df_preprocessed = self.preprocessor.transform(df_input)
        df_engineered = add_derived_features(df_preprocessed)
        
        # Ensure we drop any target column if the helper added it, and drop target column from training config
        target_col = self.config['features']['target']
        if target_col in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=[target_col])
            
        # 3. Predict probability of "Good" (Class 1)
        prob_good = float(self.model.predict_proba(df_engineered)[0, 1])
        
        # 4. Map to scores and categories
        credit_score = self.calculate_credit_score(prob_good)
        risk_cat = self.determine_risk_category(credit_score)
        recommendation = self.determine_approval_recommendation(credit_score)
        
        # 5. Save prediction history to Database
        db_prediction = PredictionHistory(
            applicant_id=db_applicant.id,
            credit_score=credit_score,
            probability_score=prob_good,
            risk_category=risk_cat,
            approval_recommendation=recommendation,
            model_version=self.model_version_str
        )
        db.add(db_prediction)
        db.commit()
        
        # 6. Construct SHAP explanation if explainer is available
        shap_explanation = {}
        if self.explainer:
            try:
                shap_explanation = self.explainer.explain_instance(df_engineered)
            except Exception as e:
                logger.error(f"Error computing SHAP values: {e}")
        
        return {
            "applicant_id": db_applicant.id,
            "credit_score": credit_score,
            "probability_score": prob_good,
            "risk_category": risk_cat,
            "approval_recommendation": recommendation,
            "shap_explanation": shap_explanation
        }

    def predict_batch_csv(self, file_path: str, db: SqlSession) -> pd.DataFrame:
        """
        Predict a batch of applicants from a CSV file.
        Returns a DataFrame with the original rows plus predictions.
        """
        if self.model is None or self.preprocessor is None:
            self.load_artifacts()
            
        df = pd.read_csv(file_path)
        
        # Create output containers
        ids = []
        scores = []
        probs = []
        risks = []
        recs = []
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            # If the csv includes creditworthiness, remove it for prediction
            target_col = self.config['features']['target']
            row_dict_clean = {k: v for k, v in row_dict.items() if k != target_col}
            
            # Predict single (this also logs to DB)
            try:
                res = self.predict_single(row_dict_clean, db)
                ids.append(res['applicant_id'])
                scores.append(res['credit_score'])
                probs.append(res['probability_score'])
                risks.append(res['risk_category'])
                recs.append(res['approval_recommendation'])
            except Exception as e:
                logger.error(f"Error predicting row {idx}: {e}")
                ids.append(None)
                scores.append(None)
                probs.append(None)
                risks.append("Prediction Error")
                recs.append("Error")
                
        df['applicant_id'] = ids
        df['predicted_credit_score'] = scores
        df['predicted_probability'] = probs
        df['risk_category'] = risks
        df['approval_recommendation'] = recs
        
        return df
