from pydantic import BaseModel, Field
from typing import Literal

class ApplicantInput(BaseModel):
    annual_income: float = Field(..., description="Annual Income", ge=0)
    monthly_income: float = Field(..., description="Monthly Income", ge=0)
    loan_amount: float = Field(..., description="Desired Loan Amount", ge=0)
    existing_debts: float = Field(..., description="Existing Debts Total", ge=0)
    debt_to_income_ratio: float = Field(..., description="Debt to Income Ratio", ge=0)
    number_of_credit_cards: int = Field(..., description="Number of Credit Cards", ge=0)
    credit_utilization_ratio: float = Field(..., description="Credit Card Utilization Ratio (0.0 to 1.0)", ge=0, le=1.5)
    payment_history: Literal["Excellent", "Good", "Fair", "Poor"] = Field(..., description="Payment History Category")
    number_of_late_payments: int = Field(..., description="Number of Late Payments", ge=0)
    loan_repayment_history: Literal["All Paid", "Mostly Paid", "Delayed", "Defaulted"] = Field(..., description="Loan Repayment History Category")
    employment_length: float = Field(..., description="Employment Length in Years", ge=0)
    age: int = Field(..., description="Age of Applicant", ge=18, le=120)
    savings_balance: float = Field(..., description="Savings Balance", ge=0)
    previous_defaults: Literal["Yes", "No"] = Field(..., description="Has Previous Defaults")
    credit_history_length: float = Field(..., description="Credit History Length in Years", ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

class PredictionResponse(BaseModel):
    credit_score: int = Field(..., description="Calculated Credit Score (300-850 range)")
    risk_category: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(..., description="Risk Category")
    approval_recommendation: Literal["Approved", "Review", "Denied"] = Field(..., description="Decision Recommendation")
    probability_score: float = Field(..., description="Probability of Good Creditworthiness")
    applicant_id: int = Field(..., description="Applicant Database Record ID")
