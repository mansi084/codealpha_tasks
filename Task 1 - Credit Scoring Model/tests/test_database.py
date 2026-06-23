import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Applicant, PredictionHistory

def test_database_operations():
    # Setup in-memory sqlite database for test
    engine = create_engine('sqlite:///:memory:')
    TestingSessionLocal = sessionmaker(bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    try:
        # Create applicant record
        applicant = Applicant(
            annual_income=75000.0,
            monthly_income=6250.0,
            loan_amount=20000.0,
            existing_debts=5000.0,
            debt_to_income_ratio=0.067,
            number_of_credit_cards=3,
            credit_utilization_ratio=0.25,
            payment_history="Excellent",
            number_of_late_payments=0,
            loan_repayment_history="All Paid",
            employment_length=6.0,
            age=32,
            savings_balance=15000.0,
            previous_defaults="No",
            credit_history_length=8.5,
            creditworthiness="Good"
        )
        db.add(applicant)
        db.commit()
        db.refresh(applicant)
        
        assert applicant.id is not None
        assert applicant.age == 32
        
        # Create prediction history record referencing applicant
        prediction = PredictionHistory(
            applicant_id=applicant.id,
            credit_score=750,
            probability_score=0.82,
            risk_category="Low Risk",
            approval_recommendation="Approved",
            model_version="v1.0.0"
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        assert prediction.id is not None
        assert prediction.applicant_id == applicant.id
        
        # Test relationship
        assert len(applicant.predictions) == 1
        assert applicant.predictions[0].credit_score == 750
        
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
