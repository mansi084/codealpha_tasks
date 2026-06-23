import os
import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings

# Suppress warnings for clean production logs
warnings.filterwarnings('ignore')

DATA_DIR = "data"
MODEL_DIR = "models"

class ExplainabilityService:
    
    @staticmethod
    def get_background_data(disease_key):
        """Loads and scales background processed dataset for training explainers."""
        df_path = os.path.join(DATA_DIR, f"{disease_key}_processed.csv")
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Processed data not found for: {disease_key}")
            
        df = pd.read_csv(df_path)
        # Find target col
        target_col = 'classification' if 'classification' in df.columns else ('Outcome' if 'Outcome' in df.columns else 'target')
        
        X = df.drop(columns=[target_col])
        features = list(X.columns)
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, f"{disease_key}_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found for: {disease_key}")
        scaler = joblib.load(scaler_path)
        
        X_scaled = scaler.transform(X)
        return X, X_scaled, features, scaler

    @classmethod
    def explain_with_lime(cls, disease_key, model, raw_input_dict):
        """
        Generates LIME explanation for a single patient record.
        Returns a dict of feature names and their impact weights.
        """
        try:
            X, X_scaled, features, scaler = cls.get_background_data(disease_key)
            
            # Map input dict to ordered features list
            ordered_values = [float(raw_input_dict[f]) for f in features]
            scaled_query = scaler.transform([ordered_values])[0]
            
            # Initialize LIME Tabular Explainer
            # Using a sample of background data to keep explanation generation fast
            background_sample = X_scaled[:150]
            
            explainer = LimeTabularExplainer(
                training_data=background_sample,
                feature_names=features,
                class_names=['Healthy', 'At Risk'],
                mode='classification',
                random_state=42
            )
            
            # Explain prediction
            exp = explainer.explain_instance(
                data_row=scaled_query,
                predict_fn=model.predict_proba,
                num_features=len(features)
            )
            
            # Parse weights
            lime_weights = {}
            for feat_idx_str, weight in exp.as_map()[1]:
                feat_name = features[int(feat_idx_str)]
                lime_weights[feat_name] = float(weight)
                
            return {
                "success": True,
                "weights": lime_weights,
                "intercept": float(exp.intercept[1])
            }
        except Exception as e:
            # Fallback local attribution
            return cls._generate_fallback_attribution(disease_key, model, raw_input_dict, f"LIME error: {str(e)}")

    @classmethod
    def explain_with_shap(cls, disease_key, model, raw_input_dict):
        """
        Generates SHAP explanation values for a single patient record.
        """
        try:
            X, X_scaled, features, scaler = cls.get_background_data(disease_key)
            
            ordered_values = [float(raw_input_dict[f]) for f in features]
            scaled_query = scaler.transform([ordered_values])
            
            # Choose appropriate SHAP explainer based on model class
            model_class = model.__class__.__name__
            
            shap_values = None
            
            # Handle Ensemble (VotingClassifier)
            if model_class == 'VotingClassifier':
                # For VotingClassifier, we calculate average shap values of underlying estimators
                shap_list = []
                for name, est in model.estimators_:
                    sub_shap = cls._get_individual_shap(est, X_scaled[:100], scaled_query)
                    if sub_shap is not None:
                        shap_list.append(sub_shap)
                if shap_list:
                    shap_values = np.mean(shap_list, axis=0)
            else:
                shap_values = cls._get_individual_shap(model, X_scaled[:100], scaled_query)
                
            if shap_values is None:
                raise ValueError("Could not calculate SHAP values")
                
            # Map features to SHAP values
            shap_dict = {features[i]: float(shap_values[0][i]) for i in range(len(features))}
            
            return {
                "success": True,
                "values": shap_dict
            }
        except Exception as e:
            return cls._generate_fallback_attribution(disease_key, model, raw_input_dict, f"SHAP error: {str(e)}")

    @staticmethod
    def _get_individual_shap(model, background, query):
        """Helper to return SHAP values for an individual estimator."""
        model_class = model.__class__.__name__
        try:
            if model_class in ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier']:
                explainer = shap.TreeExplainer(model)
                val = explainer.shap_values(query)
                # TreeExplainer output shape varies by sklearn/xgboost version
                if isinstance(val, list):
                    # For binary classifier, class 1 is index 1
                    return val[1]
                elif len(val.shape) == 3: # (samples, features, classes)
                    return val[:, :, 1]
                return val
            elif model_class == 'LogisticRegression':
                explainer = shap.LinearExplainer(model, background)
                return explainer.shap_values(query)
            else: # SVM, etc.
                # Use KernelExplainer with small background representation
                explainer = shap.KernelExplainer(model.predict_proba, background)
                val = explainer.shap_values(query)
                if isinstance(val, list):
                    return val[1]
                return val
        except Exception:
            return None

    @classmethod
    def _generate_fallback_attribution(cls, disease_key, model, raw_input_dict, reason=""):
        """Computes rule-based directionality weight attribution if SHAP/LIME fails."""
        try:
            X, X_scaled, features, scaler = cls.get_background_data(disease_key)
            ordered_values = [float(raw_input_dict[f]) for f in features]
            scaled_query = scaler.transform([ordered_values])[0]
            
            # Simple fallback using feature differences from normal/mean scaled values
            weights = {}
            for i, feat in enumerate(features):
                # Calculate how far the input is from the dataset average
                diff = scaled_query[i] - np.mean(X_scaled[:, i])
                
                # Check model coefficients direction if it exists
                direction = 1.0
                if hasattr(model, 'coef_'):
                    direction = model.coef_[0][i]
                elif hasattr(model, 'feature_importances_'):
                    direction = model.feature_importances_[i]
                
                weights[feat] = float(diff * direction * 0.1)
                
            return {
                "success": True,
                "weights": weights,
                "values": weights,
                "note": f"Fallback attribution calculated. Original error: {reason}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fallback explainer failed: {str(e)}"
            }
