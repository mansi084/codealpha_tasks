import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database.connection import Base

class PatientRecord(Base):
    __tablename__ = "patient_records"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    age = Column(Float, nullable=False)
    gender = Column(Integer, nullable=False) # 1=Male, 0=Female
    weight_kg = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    
    # Symptoms (1=Yes, 0=No)
    chest_pain = Column(Integer, default=0)
    fatigue = Column(Integer, default=0)
    frequent_urination = Column(Integer, default=0)
    shortness_of_breath = Column(Integer, default=0)
    dizziness = Column(Integer, default=0)
    nausea = Column(Integer, default=0)
    fever = Column(Integer, default=0)
    cough = Column(Integer, default=0)
    
    # Vitals
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    heart_rate = Column(Float, nullable=True)
    cholesterol = Column(Float, nullable=True)
    blood_sugar = Column(Float, nullable=True)
    oxygen_saturation = Column(Float, nullable=True)
    
    # Laboratory Results
    glucose = Column(Float, nullable=True)
    hba1c = Column(Float, nullable=True)
    insulin = Column(Float, nullable=True)
    creatinine = Column(Float, nullable=True)
    wbc_count = Column(Float, nullable=True)
    rbc_count = Column(Float, nullable=True)
    platelet_count = Column(Float, nullable=True)
    
    # Medical History (1=Yes, 0=No)
    smoking_status = Column(Integer, default=0)
    alcohol_consumption = Column(Integer, default=0)
    family_history = Column(Integer, default=0)
    previous_diseases = Column(Text, nullable=True)
    medication_history = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    predictions = relationship("PredictionResult", back_populates="patient", cascade="all, delete-orphan")

class PredictionResult(Base):
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patient_records.id", ondelete="CASCADE"), nullable=False)
    prediction_date = Column(DateTime, default=datetime.datetime.utcnow)
    disease_type = Column(String(50), nullable=False) # e.g. 'heart_disease', 'diabetes'
    probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False) # 'Low', 'Moderate', 'High'
    confidence = Column(Float, default=0.90)
    contributing_factors = Column(Text, nullable=True) # JSON representation of feature impacts
    recommendation = Column(Text, nullable=True)
    
    patient = relationship("PatientRecord", back_populates="predictions")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    disease_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    algorithm = Column(String(50), nullable=False)
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    training_date = Column(DateTime, default=datetime.datetime.utcnow)
    file_path = Column(String(255), nullable=True)

class UserLog(Base):
    __tablename__ = "user_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    details = Column(Text, nullable=True)
