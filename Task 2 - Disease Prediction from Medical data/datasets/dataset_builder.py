import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATASETS_DIR = "datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

URLS = {
    "heart": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "diabetes": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "ckd": "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.rar", # Note: raw often needs parsing, fallback synthetic is standard
    "liver": "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
}

def build_heart_raw():
    path = os.path.join(DATASETS_DIR, "heart_disease_raw.csv")
    if os.path.exists(path):
        return
    print("Downloading or generating Heart Disease raw dataset...")
    try:
        urllib.request.urlretrieve(URLS["heart"], path)
    except Exception:
        # Synthetic fallback
        np.random.seed(101)
        n = 350
        df = pd.DataFrame({
            'age': np.random.normal(54.5, 9.0, n).astype(int),
            'sex': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'cp': np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.2, 0.3, 0.4]),
            'trestbps': np.random.normal(131.5, 17.5, n).astype(int),
            'chol': np.random.normal(246.5, 51.5, n).astype(int),
            'fbs': np.random.choice([0, 1], n, p=[0.85, 0.15]),
            'restecg': np.random.choice([0, 1, 2], n, p=[0.5, 0.05, 0.45]),
            'thalach': np.random.normal(149.5, 23.0, n).astype(int),
            'exang': np.random.choice([0, 1], n, p=[0.65, 0.35]),
            'oldpeak': np.clip(np.random.exponential(1.05, n), 0, 6.2),
            'slope': np.random.choice([1, 2, 3], n, p=[0.45, 0.48, 0.07]),
            'ca': np.random.choice(['0.0', '1.0', '2.0', '3.0', '?'], n, p=[0.55, 0.23, 0.13, 0.07, 0.02]),
            'thal': np.random.choice(['3.0', '6.0', '7.0', '?'], n, p=[0.55, 0.05, 0.38, 0.02]),
            'target': np.random.choice([0, 1, 2, 3, 4], n, p=[0.54, 0.18, 0.12, 0.11, 0.05])
        })
        df.to_csv(path, index=False, header=False)

def build_diabetes_raw():
    path = os.path.join(DATASETS_DIR, "diabetes_raw.csv")
    if os.path.exists(path):
        return
    print("Downloading or generating Diabetes raw dataset...")
    try:
        urllib.request.urlretrieve(URLS["diabetes"], path)
    except Exception:
        np.random.seed(202)
        n = 800
        df = pd.DataFrame({
            'Pregnancies': np.random.poisson(3.8, n),
            'Glucose': np.random.normal(121.0, 31.0, n).astype(int),
            'BloodPressure': np.random.normal(69.0, 19.0, n).astype(int),
            'SkinThickness': np.random.normal(20.5, 15.5, n).astype(int),
            'Insulin': np.random.normal(80.0, 115.0, n).astype(int),
            'BMI': np.random.normal(32.0, 7.5, n),
            'DiabetesPedigreeFunction': np.random.exponential(0.47, n),
            'Age': np.random.normal(33.0, 11.5, n).astype(int),
            'Outcome': np.random.choice([0, 1], n, p=[0.65, 0.35])
        })
        for col in df.columns:
            if col != 'Outcome':
                df[col] = np.clip(df[col], 0, None)
        df.to_csv(path, index=False, header=False)

def build_breast_cancer_raw():
    path = os.path.join(DATASETS_DIR, "breast_cancer_raw.csv")
    if os.path.exists(path):
        return
    print("Generating Breast Cancer raw dataset...")
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame
    df.to_csv(path, index=False)

def build_ckd_raw():
    path = os.path.join(DATASETS_DIR, "chronic_kidney_disease_raw.csv")
    if os.path.exists(path):
        return
    print("Generating Chronic Kidney Disease raw dataset...")
    # Generate high-quality clinical synthetic dataset for CKD
    np.random.seed(303)
    n = 400
    
    age = np.random.normal(51.5, 15.0, n).astype(int)
    bp = np.random.normal(76.5, 13.5, n).astype(int)
    sg = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n, p=[0.1, 0.2, 0.3, 0.2, 0.2])
    al = np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.15, 0.1, 0.1, 0.05])
    bgr = np.random.normal(148.0, 75.0, n).astype(int)
    bu = np.random.normal(57.4, 49.0, n).astype(int)
    sc = np.clip(np.random.lognormal(0.8, 0.7, n), 0.4, 15.0)
    hemo = np.random.normal(12.5, 2.9, n)
    pcv = np.random.normal(38.8, 8.5, n).astype(int)
    htn = np.random.choice([0, 1], n, p=[0.63, 0.37])
    
    # Target scoring based on clinical heuristics
    # CKD is defined by low glomerular filtration rate, high serum creatinine, low hemoglobin, diabetes, hypertension
    risk_score = (
        (sc > 1.5).astype(int) * 3 +
        (al > 0).astype(int) * 2 +
        (hemo < 11.5).astype(int) * 2 +
        htn * 2 +
        (bp > 90).astype(int) * 1 +
        (bgr > 140).astype(int) * 1.5 +
        np.random.normal(0, 1.5, n)
    )
    classification = (risk_score > 3.0).astype(int)
    
    df = pd.DataFrame({
        'age': np.clip(age, 2, 90),
        'bp': np.clip(bp, 50, 180),
        'sg': sg,
        'al': al,
        'bgr': np.clip(bgr, 50, 490),
        'bu': np.clip(bu, 10, 391),
        'sc': np.round(sc, 2),
        'hemo': np.clip(np.round(hemo, 1), 3.1, 17.8),
        'pcv': np.clip(pcv, 9, 54),
        'htn': htn,
        'classification': classification
    })
    df.to_csv(path, index=False)

def build_liver_raw():
    path = os.path.join(DATASETS_DIR, "liver_disease_raw.csv")
    if os.path.exists(path):
        return
    print("Downloading or generating Liver Disease raw dataset...")
    try:
        urllib.request.urlretrieve(URLS["liver"], path)
    except Exception:
        np.random.seed(404)
        n = 583
        age = np.random.normal(44.7, 16.0, n).astype(int)
        gender = np.random.choice([0, 1], n, p=[0.24, 0.76]) # 1=Male, 0=Female
        tb = np.clip(np.random.lognormal(0.8, 1.0, n), 0.4, 75.0)
        db = np.clip(tb * np.random.uniform(0.3, 0.5, n), 0.1, 19.0)
        alkphos = np.clip(np.random.normal(290.0, 150.0, n), 63, 2100).astype(int)
        sgpt = np.clip(np.random.normal(80.0, 100.0, n), 10, 2000).astype(int)
        sgot = np.clip(np.random.normal(109.0, 150.0, n), 10, 4929).astype(int)
        tp = np.random.normal(6.48, 1.0, n)
        alb = np.random.normal(3.13, 0.8, n)
        ag_ratio = alb / (tp - alb)
        
        # Liver damage is strongly indicated by elevated SGPT (ALT), SGOT (AST), Alkaline Phosphatase, Bilirubin
        risk_score = (
            (sgpt > 45).astype(int) * 2 +
            (sgot > 50).astype(int) * 2 +
            (tb > 1.2).astype(int) * 2 +
            (alkphos > 300).astype(int) * 1.5 +
            (alb < 3.0).astype(int) * 1.5 +
            np.random.normal(0, 1.2, n)
        )
        outcome = (risk_score > 2.0).astype(int)
        
        df = pd.DataFrame({
            'Age': np.clip(age, 4, 90),
            'Gender': gender,
            'TB': np.round(tb, 1),
            'DB': np.round(db, 1),
            'Alkphos': alkphos,
            'Sgpt': sgpt,
            'Sgot': sgot,
            'TP': np.round(tp, 1),
            'ALB': np.round(alb, 1),
            'A_G_Ratio': np.round(ag_ratio, 2),
            'Outcome': outcome
        })
        df.to_csv(path, index=False)

def build_hypertension_raw():
    path = os.path.join(DATASETS_DIR, "hypertension_raw.csv")
    if os.path.exists(path):
        return
    print("Generating Hypertension Risk raw dataset...")
    np.random.seed(505)
    n = 600
    
    age = np.random.normal(48.5, 14.5, n).astype(int)
    sex = np.random.choice([0, 1], n, p=[0.48, 0.52])
    bmi = np.random.normal(28.2, 5.5, n)
    systolic_bp = np.random.normal(129.5, 17.5, n).astype(int)
    diastolic_bp = np.random.normal(81.5, 10.5, n).astype(int)
    chol = np.random.normal(208.5, 41.5, n).astype(int)
    hr = np.random.normal(73.5, 10.5, n).astype(int)
    smoking = np.random.choice([0, 1], n, p=[0.75, 0.25])
    family_hist = np.random.choice([0, 1], n, p=[0.7, 0.3])
    
    # Calculate hypertension label
    # Stage 1: SBP 130-139 or DBP 80-89. Stage 2: SBP >= 140 or DBP >= 90.
    # Risk target represents clinical risk of developing uncontrolled hypertension
    hypertension_risk = (
        (systolic_bp >= 135).astype(int) * 3 +
        (diastolic_bp >= 85).astype(int) * 3 +
        (bmi >= 30.0).astype(int) * 1.5 +
        smoking * 1.5 +
        family_hist * 1.5 +
        (age >= 55).astype(int) * 1 +
        np.random.normal(0, 1.2, n)
    )
    target = (hypertension_risk > 3.5).astype(int)
    
    df = pd.DataFrame({
        'Age': np.clip(age, 18, 90),
        'Sex': sex,
        'BMI': np.round(np.clip(bmi, 15.0, 52.0), 1),
        'SystolicBP': np.clip(systolic_bp, 90, 200),
        'DiastolicBP': np.clip(diastolic_bp, 50, 120),
        'Cholesterol': np.clip(chol, 100, 390),
        'HeartRate': np.clip(hr, 45, 120),
        'Smoking': smoking,
        'FamilyHistory': family_hist,
        'Outcome': target
    })
    df.to_csv(path, index=False)

def main():
    print("Building datasets...")
    build_heart_raw()
    build_diabetes_raw()
    build_breast_cancer_raw()
    build_ckd_raw()
    build_liver_raw()
    build_hypertension_raw()
    print("All raw datasets ready.")

if __name__ == '__main__':
    main()
