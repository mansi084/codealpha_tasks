import os
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Local imports
from database.db import Base, get_db
from backend.api import app

# Setup testing db
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_credit_scoring.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Override dependency
app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    # Dispose engine to release sqlite file lock on Windows
    engine.dispose()
    # Remove database file
    if os.path.exists("./test_credit_scoring.db"):
        os.remove("./test_credit_scoring.db")

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "annual_income": 75000.0,
        "monthly_income": 6250.0,
        "loan_amount": 25000.0,
        "existing_debts": 12000.0,
        "debt_to_income_ratio": 0.16,
        "number_of_credit_cards": 3,
        "credit_utilization_ratio": 0.28,
        "payment_history": "Excellent",
        "number_of_late_payments": 0,
        "loan_repayment_history": "All Paid",
        "employment_length": 5.5,
        "age": 34,
        "savings_balance": 18500.0,
        "previous_defaults": "No",
        "credit_history_length": 10.2
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "credit_score" in data
    assert "risk_category" in data
    assert "approval_recommendation" in data
    assert "probability_score" in data
    assert "applicant_id" in data
    
    # Assert values range
    assert 300 <= data["credit_score"] <= 850
    assert data["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert data["approval_recommendation"] in ["Approved", "Review", "Denied"]

def test_model_metrics_endpoint():
    response = client.get("/model-metrics")
    assert response.status_code == 200
    # It should return a list
    assert isinstance(response.json(), list)

def test_feature_importance_endpoint():
    response = client.get("/feature-importance")
    # If the file exists, it will return 200. Let's assert either 200 or 404 (handled gracefully)
    assert response.status_code in [200, 404]

def test_batch_predict_endpoint(tmp_path):
    # Create dummy csv file
    csv_file = tmp_path / "batch_test.csv"
    df = pd_dummy = pd_data = """annual_income,monthly_income,loan_amount,existing_debts,debt_to_income_ratio,number_of_credit_cards,credit_utilization_ratio,payment_history,number_of_late_payments,loan_repayment_history,employment_length,age,savings_balance,previous_defaults,credit_history_length
75000.0,6250.0,25000.0,12000.0,0.16,3,0.28,Excellent,0,All Paid,5.5,34,18500.0,No,10.2
50000.0,4166.0,10000.0,2000.0,0.04,2,0.15,Good,0,Mostly Paid,3.0,29,5000.0,No,5.0
"""
    csv_file.write_text(df)
    
    with open(csv_file, "rb") as f:
        response = client.post("/batch-predict", files={"file": ("batch_test.csv", f, "text/csv")})
        
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert "predicted_credit_score" in data[0]
    assert "risk_category" in data[0]
