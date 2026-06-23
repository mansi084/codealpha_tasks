import os
import json
import unittest
from fastapi.testclient import TestClient

# Make sure imports search root directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api import app
from database.connection import SessionLocal
from database.models import PatientRecord, PredictionResult

class TestClinicalDecisionSupportSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        
    def test_health_check_endpoints(self):
        """Verify that basic metadata endpoints load successfully."""
        # 1. Model metrics endpoint
        response = self.client.get("/model-metrics")
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertIn("heart_disease", res_data)
        self.assertIn("diabetes", res_data)
        self.assertIn("hypertension", res_data)
        
        # 2. Feature importance endpoint
        response = self.client.get("/feature-importance")
        self.assertEqual(response.status_code, 200)
        self.assertIn("heart_disease", response.json())

    def test_single_disease_predictions(self):
        """Verify that individual disease prediction endpoints process and score correctly."""
        # Heart disease endpoint test
        heart_payload = {
            "name": "Jane HeartTest", "age": 60.0, "gender": 1, "weight_kg": 85.0, "height_cm": 175.0,
            "systolic_bp": 145.0, "diastolic_bp": 92.0, "glucose": 110.0, "cholesterol": 260.0,
            "heart_rate": 82.0, "chest_pain": 4, "smoking_status": 1, "family_history": 1
        }
        response = self.client.post("/predict/heart-disease", json=heart_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["disease_name"], "Heart Disease Prediction")
        self.assertIn("probability", data)
        self.assertIn("risk_level", data)
        self.assertIn("contributing_factors", data)
        
        # Diabetes endpoint test
        diabetes_payload = {
            "name": "John DiaTest", "age": 42.0, "gender": 0, "weight_kg": 95.0, "height_cm": 160.0,
            "glucose": 158.0, "hba1c": 6.8, "insulin": 120.0, "family_history": 1
        }
        response = self.client.post("/predict/diabetes", json=diabetes_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["disease_name"], "Diabetes Risk Prediction")
        self.assertGreater(data["probability"], 0.0)

    def test_multi_disease_and_db_logging(self):
        """Verify unified /predict/all runs 6 models, registers patient and logs to DB."""
        payload = {
            "name": "Alice UnifiedTest",
            "age": 55.0,
            "gender": 0,
            "weight_kg": 72.0,
            "height_cm": 165.0,
            "systolic_bp": 138.0,
            "diastolic_bp": 85.0,
            "heart_rate": 74.0,
            "glucose": 128.0,
            "hba1c": 6.1,
            "creatinine": 1.4,
            "cholesterol": 235.0,
            "chest_pain": 0,
            "fatigue": 1,
            "smoking_status": 0,
            "family_history": 1
        }
        response = self.client.post("/predict/all", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["patient_name"], "Alice UnifiedTest")
        self.assertIn("composite_health_score", data)
        self.assertEqual(len(data["predictions"]), 6)
        
        # Verify it was committed to SQL local database
        db = SessionLocal()
        db_patient = db.query(PatientRecord).filter_by(name="Alice UnifiedTest").order_by(PatientRecord.created_at.desc()).first()
        self.assertIsNotNone(db_patient)
        self.assertEqual(db_patient.age, 55.0)
        
        # Check prediction results link
        db_preds = db.query(PredictionResult).filter_by(patient_id=db_patient.id).all()
        self.assertEqual(len(db_preds), 6)
        db.close()

    def test_rag_chatbot(self):
        """Verify matching queries against text guidelines."""
        # 1. Ask about high blood pressure guidelines
        chat_payload = {
            "message": "What is the recommended diet for a patient with high blood pressure?"
        }
        response = self.client.post("/chat", json=chat_payload)
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertIn("DASH", res_data["response"])
        self.assertIn("Hypertension", res_data["response"])
        
        # 2. Ask with a symptom checking query
        chat_payload_symptom = {
            "message": "My patient has sudden chest pain and shortness of breath"
        }
        response = self.client.post("/chat", json=chat_payload_symptom)
        self.assertEqual(response.status_code, 200)
        res_data_symptom = response.json()
        self.assertIn("Chest Pain", res_data_symptom["response"])
        self.assertIn("emergency", res_data_symptom["response"].lower())

    def test_batch_predict(self):
        """Verify CSV batch uploader matches columns and processes lines."""
        csv_data = (
            "name,age,gender,weight_kg,height_cm,glucose,systolic_bp,diastolic_bp,cholesterol,heart_rate,smoking_status,family_history\n"
            "TestPatientA,45,0,65,160,98,125,82,192,72,0,1\n"
            "TestPatientB,62,1,85,175,145,142,88,245,80,1,1\n"
        )
        response = self.client.post(
            "/batch-predict",
            files={"file": ("test_patients.csv", csv_data, "text/csv")}
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertTrue(res_data["success"])
        self.assertEqual(res_data["processed_count"], 2)
        self.assertEqual(len(res_data["predictions"]), 2)

if __name__ == '__main__':
    unittest.main()
