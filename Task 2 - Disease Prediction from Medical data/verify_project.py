import os
import json
import joblib
import unittest
import numpy as np
from app import app, get_model_and_scaler

class TestDiseasePredictionSystem(unittest.TestCase):
    
    def test_model_files_exist(self):
        """Verify that all 12 models and 3 scalers are trained and serialized."""
        diseases = ['heart_disease', 'diabetes', 'breast_cancer']
        algorithms = ['logistic_regression', 'svm', 'random_forest', 'xgboost']
        
        for disease in diseases:
            # Check scaler
            scaler_path = f"models/{disease}_scaler.joblib"
            self.assertTrue(os.path.exists(scaler_path), f"Scaler not found: {scaler_path}")
            
            # Check models
            for algo in algorithms:
                model_path = f"models/{disease}_{algo}.joblib"
                self.assertTrue(os.path.exists(model_path), f"Model not found: {model_path}")

    def test_metrics_json(self):
        """Verify that model_metrics.json is correctly generated and parsed."""
        metrics_path = 'static/model_metrics.json'
        self.assertTrue(os.path.exists(metrics_path), f"Metrics JSON not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        diseases = ['heart_disease', 'diabetes', 'breast_cancer']
        algorithms = ['logistic_regression', 'svm', 'random_forest', 'xgboost']
        
        for disease in diseases:
            self.assertIn(disease, metrics)
            self.assertIn('features', metrics[disease])
            self.assertIn('models', metrics[disease])
            
            for algo in algorithms:
                self.assertIn(algo, metrics[disease]['models'])
                model_metrics = metrics[disease]['models'][algo]
                
                # Check metrics are positive floats
                self.assertGreaterEqual(model_metrics['accuracy'], 0.6) # should be at least >60%
                self.assertGreaterEqual(model_metrics['roc_auc'], 0.7)  # should be at least >70%
                self.assertIn('confusion_matrix', model_metrics)
                self.assertIn('roc_curve', model_metrics)

    def test_flask_predict_api(self):
        """Mock the Flask test client and test prediction endpoints."""
        client = app.test_client()
        
        # Test 1: Heart Disease Prediction (At risk sample)
        heart_payload = {
            "disease": "heart_disease",
            "algorithm": "random_forest",
            "features": {
                "age": 64.0, "sex": 1.0, "cp": 4.0, "trestbps": 152.0, "chol": 294.0, 
                "fbs": 1.0, "restecg": 2.0, "thalach": 108.0, "exang": 1.0, 
                "oldpeak": 2.8, "slope": 2.0, "ca": 3.0, "thal": 7.0
            }
        }
        response = client.post('/api/predict', 
                               data=json.dumps(heart_payload), 
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        self.assertTrue(res_data['success'])
        self.assertIn('prediction', res_data)
        self.assertIn('probability', res_data)
        self.assertIn('risk_level', res_data)
        
        # Test 2: Diabetes Prediction (Healthy sample)
        diabetes_payload = {
            "disease": "diabetes",
            "algorithm": "logistic_regression",
            "features": {
                "Pregnancies": 1.0, "Glucose": 92.0, "BloodPressure": 64.0, 
                "SkinThickness": 18.0, "Insulin": 72.0, "BMI": 21.8, 
                "DiabetesPedigreeFunction": 0.22, "Age": 24.0
            }
        }
        response = client.post('/api/predict', 
                               data=json.dumps(diabetes_payload), 
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        self.assertTrue(res_data['success'])
        
        # Test 3: Breast Cancer (Invalid payload features missing)
        invalid_payload = {
            "disease": "breast_cancer",
            "algorithm": "xgboost",
            "features": {
                "mean radius": 14.1 # missing other 9 features
            }
        }
        response = client.post('/api/predict', 
                               data=json.dumps(invalid_payload), 
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        res_data = json.loads(response.data)
        self.assertFalse(res_data['success'])
        self.assertIn('error', res_data)

if __name__ == '__main__':
    unittest.main()
