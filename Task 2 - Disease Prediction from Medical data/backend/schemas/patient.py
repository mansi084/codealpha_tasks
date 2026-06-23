from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PatientIntakeSchema(BaseModel):
    name: str = Field(..., example="John Doe")
    age: float = Field(..., ge=0, le=120, example=55.0)
    gender: int = Field(..., ge=0, le=1, description="1 = Male, 0 = Female", example=1)
    weight_kg: Optional[float] = Field(None, ge=10, le=250, example=82.0)
    height_cm: Optional[float] = Field(None, ge=50, le=250, example=178.0)
    pregnancies: Optional[int] = Field(0, ge=0, le=20, example=0)
    
    # Symptoms (0 = None, 1 = Present / Mild, 2 = Severe)
    chest_pain: Optional[int] = Field(0, ge=0, le=4, description="0=None, 1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic", example=0)
    fatigue: Optional[int] = Field(0, ge=0, le=1, example=0)
    frequent_urination: Optional[int] = Field(0, ge=0, le=1, example=0)
    shortness_of_breath: Optional[int] = Field(0, ge=0, le=1, example=0)
    dizziness: Optional[int] = Field(0, ge=0, le=1, example=0)
    nausea: Optional[int] = Field(0, ge=0, le=1, example=0)
    fever: Optional[int] = Field(0, ge=0, le=1, example=0)
    cough: Optional[int] = Field(0, ge=0, le=1, example=0)
    
    # Vitals / Clinical Measurements
    systolic_bp: Optional[float] = Field(120.0, ge=50, le=250, example=135.0)
    diastolic_bp: Optional[float] = Field(80.0, ge=30, le=150, example=85.0)
    heart_rate: Optional[float] = Field(72.0, ge=30, le=200, example=78.0)
    cholesterol: Optional[float] = Field(190.0, ge=80, le=500, example=220.0)
    oxygen_saturation: Optional[float] = Field(98.0, ge=50, le=100, example=97.0)
    
    # Laboratory Results
    glucose: Optional[float] = Field(95.0, ge=40, le=600, example=110.0)
    hba1c: Optional[float] = Field(5.4, ge=3.0, le=18.0, example=5.8)
    insulin: Optional[float] = Field(80.0, ge=5, le=1000, example=95.0)
    creatinine: Optional[float] = Field(0.9, ge=0.1, le=30.0, example=1.1)
    wbc_count: Optional[float] = Field(6000.0, ge=1000, le=50000, example=7200.0)
    rbc_count: Optional[float] = Field(4.8, ge=1.0, le=10.0, example=4.5)
    platelet_count: Optional[float] = Field(250000.0, ge=10000, le=1000000, example=260000.0)
    
    # Laboratory parameters for breast cancer morphology (Wisconsin attributes)
    worst_area: Optional[float] = Field(880.0, description="Wisconsin Breast Cancer - Worst Area parameter", example=880.0)
    worst_concave_points: Optional[float] = Field(0.12, description="Wisconsin Breast Cancer - Worst Concave Points", example=0.12)
    mean_concave_points: Optional[float] = Field(0.05, description="Wisconsin Breast Cancer - Mean Concave Points", example=0.05)
    worst_radius: Optional[float] = Field(16.0, description="Wisconsin Breast Cancer - Worst Radius", example=16.0)
    worst_perimeter: Optional[float] = Field(107.0, description="Wisconsin Breast Cancer - Worst Perimeter", example=107.0)
    mean_perimeter: Optional[float] = Field(92.0, description="Wisconsin Breast Cancer - Mean Perimeter", example=92.0)
    mean_concavity: Optional[float] = Field(0.09, description="Wisconsin Breast Cancer - Mean Concavity", example=0.09)
    mean_area: Optional[float] = Field(650.0, description="Wisconsin Breast Cancer - Mean Area", example=650.0)
    worst_concavity: Optional[float] = Field(0.27, description="Wisconsin Breast Cancer - Worst Concavity", example=0.27)
    mean_radius: Optional[float] = Field(14.1, description="Wisconsin Breast Cancer - Mean Radius", example=14.1)

    # Medical History
    smoking_status: Optional[int] = Field(0, ge=0, le=1, example=0)
    alcohol_consumption: Optional[int] = Field(0, ge=0, le=1, example=0)
    family_history: Optional[int] = Field(0, ge=0, le=1, example=0)
    previous_diseases: Optional[str] = Field("", example="None")
    medication_history: Optional[str] = Field("", example="None")

class FactorImpact(BaseModel):
    feature: str
    impact: float
    description: str

class DiseasePredictionResponse(BaseModel):
    disease_name: str
    probability: float
    risk_level: str
    confidence: float
    contributing_factors: List[FactorImpact]
    recommendation: str

class MultiDiseasePredictionResponse(BaseModel):
    success: bool
    patient_name: str
    bmi: float
    bmi_category: str
    bp_category: str
    composite_health_score: float
    predictions: List[DiseasePredictionResponse]

class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
