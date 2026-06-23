import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Applicant(Base):
    __tablename__ = 'applicants'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    annual_income = Column(Float, nullable=False)
    monthly_income = Column(Float, nullable=False)
    loan_amount = Column(Float, nullable=False)
    existing_debts = Column(Float, nullable=False)
    debt_to_income_ratio = Column(Float, nullable=False)
    number_of_credit_cards = Column(Integer, nullable=False)
    credit_utilization_ratio = Column(Float, nullable=False)
    payment_history = Column(String(50), nullable=False)        # Excellent, Good, Fair, Poor
    number_of_late_payments = Column(Integer, nullable=False)
    loan_repayment_history = Column(String(50), nullable=False) # All Paid, Mostly Paid, Delayed, Defaulted
    employment_length = Column(Float, nullable=False)          # in years
    age = Column(Integer, nullable=False)
    savings_balance = Column(Float, nullable=False)
    previous_defaults = Column(String(10), nullable=False)      # Yes, No
    credit_history_length = Column(Float, nullable=False)      # in years
    creditworthiness = Column(String(10), nullable=True)        # Good, Bad (used if it's ground truth)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    predictions = relationship("PredictionHistory", back_populates="applicant", cascade="all, delete-orphan")

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    applicant_id = Column(Integer, ForeignKey('applicants.id'), nullable=False)
    credit_score = Column(Integer, nullable=False)             # e.g., 300 to 850
    probability_score = Column(Float, nullable=False)          # Probability of "Good"
    risk_category = Column(String(50), nullable=False)         # Low Risk, Medium Risk, High Risk
    approval_recommendation = Column(String(50), nullable=False) # Approved, Review, Denied
    model_version = Column(String(50), nullable=False)
    predicted_at = Column(DateTime, default=datetime.datetime.utcnow)

    applicant = relationship("Applicant", back_populates="predictions")

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False)
    model_name = Column(String(100), nullable=False)
    file_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=False)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)

    metrics = relationship("EvaluationMetric", back_populates="model", cascade="all, delete-orphan")

class EvaluationMetric(Base):
    __tablename__ = 'evaluation_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), ForeignKey('model_versions.version'), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)

    model = relationship("ModelVersion", back_populates="metrics")

class UserActivityLog(Base):
    __tablename__ = 'user_activity_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), default="system")
    action = Column(String(255), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
