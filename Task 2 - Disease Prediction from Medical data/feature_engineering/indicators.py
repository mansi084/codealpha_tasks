import numpy as np
import pandas as pd

def calculate_bmi(weight_kg, height_cm):
    """Computes BMI given weight in kilograms and height in centimeters."""
    if height_cm <= 0:
        return 0.0
    height_m = height_cm / 100.0
    return round(float(weight_kg / (height_m ** 2)), 2)

def get_bmi_category(bmi):
    """Categorizes BMI value into standard WHO categories."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal weight"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"

def get_bp_category(systolic, diastolic):
    """Determines American Heart Association (AHA) blood pressure category."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic < 130 and diastolic < 80:
        return "Elevated"
    elif 130 <= systolic < 140 or 80 <= diastolic < 90:
        return "Hypertension Stage 1"
    elif 140 <= systolic <= 180 or 90 <= diastolic <= 120:
        return "Hypertension Stage 2"
    elif systolic > 180 or diastolic > 120:
        return "Hypertensive Crisis"
    else:
        return "Normal"  # Fallback

def calculate_diabetes_risk_score(glucose, bmi, age, family_history_diabetes):
    """
    Computes a simplified clinical risk score for diabetes (0-10).
    Based on ADA guidelines.
    """
    score = 0
    # Age factor
    if age > 65:
        score += 3
    elif age > 45:
        score += 2
    
    # BMI factor
    if bmi >= 30:
        score += 3
    elif bmi >= 25:
        score += 1
        
    # Glucose factor (fasting or random)
    if glucose > 125:
        score += 3
    elif glucose > 100:
        score += 1
        
    # Family history factor
    if family_history_diabetes:
        score += 1
        
    return score

def calculate_cardiac_risk_score(systolic_bp, cholesterol, age, smoking_status, family_history_heart):
    """
    Computes a cardiac risk score (0-10) using clinical risk attributes.
    """
    score = 0
    # Age factor
    if age > 60:
        score += 3
    elif age > 45:
        score += 1
        
    # BP factor
    if systolic_bp >= 140:
        score += 3
    elif systolic_bp >= 130:
        score += 1
        
    # Cholesterol factor
    if cholesterol >= 240:
        score += 2
    elif cholesterol >= 200:
        score += 1
        
    # Smoking status
    if smoking_status:
        score += 2
        
    # Family history
    if family_history_heart:
        score += 1
        
    return score

def calculate_cholesterol_ratio(total_cholesterol, hdl_cholesterol=None):
    """Computes total cholesterol to HDL cholesterol ratio. Fallback if HDL is missing."""
    if hdl_cholesterol and hdl_cholesterol > 0:
        return round(float(total_cholesterol / hdl_cholesterol), 2)
    # Estimate HDL at ~50 mg/dL if missing
    return round(float(total_cholesterol / 50.0), 2)

def calculate_lifestyle_risk_score(smoking, alcohol, bmi, physical_activity=1):
    """
    Computes a lifestyle risk score (0-4).
    physical_activity: 1 for active, 0 for sedentary.
    """
    score = 0
    if smoking:
        score += 1
    if alcohol:
        score += 1
    if bmi >= 30.0:
        score += 1
    if physical_activity == 0:
        score += 1
    return score

def calculate_family_history_index(family_diabetes, family_heart, family_cancer):
    """Computes family history index score (0-3)."""
    score = 0
    if family_diabetes:
        score += 1
    if family_heart:
        score += 1
    if family_cancer:
        score += 1
    return score

def calculate_composite_health_score(patient_dict):
    """
    Computes a composite health score from 0 to 100,
    where 100 is optimal and deductions are made for risk variables.
    """
    score = 100
    
    # Demographics / Vitals
    age = patient_dict.get('age', 30)
    bmi = patient_dict.get('bmi', 22.0)
    systolic = patient_dict.get('systolic_bp', 120)
    diastolic = patient_dict.get('diastolic_bp', 80)
    glucose = patient_dict.get('glucose', 90)
    chol = patient_dict.get('cholesterol', 180)
    
    # History
    smoking = patient_dict.get('smoking', 0)
    alcohol = patient_dict.get('alcohol', 0)
    
    # Deductions
    # BP Deductions
    if systolic >= 140 or diastolic >= 90:
        score -= 15
    elif systolic >= 130 or diastolic >= 80:
        score -= 8
        
    # Glucose Deductions
    if glucose >= 126:
        score -= 15
    elif glucose >= 100:
        score -= 5
        
    # Cholesterol Deductions
    if chol >= 240:
        score -= 10
    elif chol >= 200:
        score -= 4
        
    # BMI Deductions
    if bmi >= 30:
        score -= 12
    elif bmi >= 25:
        score -= 5
    elif bmi < 18.5:
        score -= 5
        
    # Lifestyle
    if smoking:
        score -= 15
    if alcohol:
        score -= 8
        
    # Age adjustments (gradual penalty for natural vascular aging, cap deduction)
    if age > 65:
        score -= 10
    elif age > 50:
        score -= 5
        
    return max(0, min(100, score))

def add_indicators_to_dataframe(df, disease_type):
    """Helper to inject computed indicators into training or evaluation DataFrames."""
    df_feat = df.copy()
    if disease_type == 'heart_disease':
        # Mapping: trestbps=systolic, chol=cholesterol, exang=lifestyle proxy
        df_feat['cholesterol_ratio'] = df_feat['chol'].apply(lambda x: calculate_cholesterol_ratio(x))
        df_feat['cardiac_risk_score'] = df_feat.apply(
            lambda r: calculate_cardiac_risk_score(r['trestbps'], r['chol'], r['age'], r['exang'], 0), axis=1
        )
    elif disease_type == 'diabetes':
        # Mapping: Glucose, BMI, Age, DiabetesPedigreeFunction
        df_feat['diabetes_risk_score'] = df_feat.apply(
            lambda r: calculate_diabetes_risk_score(r['Glucose'], r['BMI'], r['Age'], r['DiabetesPedigreeFunction'] > 0.5), axis=1
        )
    elif disease_type == 'hypertension':
        df_feat['bp_category'] = df_feat.apply(
            lambda r: get_bp_category(r['SystolicBP'], r['DiastolicBP']), axis=1
        )
        df_feat['cardiac_risk_score'] = df_feat.apply(
            lambda r: calculate_cardiac_risk_score(r['SystolicBP'], r['Cholesterol'], r['Age'], r['Smoking'], r['FamilyHistory']), axis=1
        )
    return df_feat
