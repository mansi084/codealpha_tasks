import os
import numpy as np
import pandas as pd

RAW_DIR = "datasets"
PROCESSED_DIR = "data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Physiological bounds for medical variables to ensure data integrity
PHYSIOLOGICAL_BOUNDS = {
    'age': (0, 125),
    'Age': (0, 125),
    'systolic': (40, 300),
    'diastolic': (20, 200),
    'bmi': (10, 80),
    'BMI': (10, 80),
    'glucose': (20, 800),
    'Glucose': (20, 800),
    'cholesterol': (50, 800),
    'Cholesterol': (50, 800),
    'heart_rate': (30, 220),
    'HeartRate': (30, 220),
    'temperature': (94, 110),
    'oxygen': (50, 100)
}

def validate_and_clean_column(df, column, val_min, val_max, fallback_median=True):
    """Checks physiological validity of columns, replacing invalid numbers with NaN or median."""
    if column not in df.columns:
        return df
    
    # Replace outliers or zeroes that are physically impossible
    # In some datasets like Pima, 0 is used for missing blood pressure, insulin, skin thickness, etc.
    invalid_mask = (df[column] < val_min) | (df[column] > val_max)
    
    # Replace with NaN to leverage imputation
    df.loc[invalid_mask, column] = np.nan
    
    if fallback_median:
        median_val = df[column].median()
        if pd.isna(median_val):
            median_val = (val_min + val_max) / 2.0
        df[column] = df[column].fillna(median_val)
        
    return df

def impute_missing_values(df):
    """Imputes missing numbers with median, and categorical columns with mode."""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    return df

def handle_outliers_iqr(df, exclude_cols=None):
    """Detects and clips outliers to 1.5 * IQR to avoid skewed model fits."""
    if exclude_cols is None:
        exclude_cols = ['target', 'Outcome', 'classification']
        
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if col in exclude_cols:
            continue
            
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Clip outliers instead of dropping them, which preserves dataset volume
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
    return df_clean

def preprocess_heart_disease():
    raw_path = os.path.join(RAW_DIR, "heart_disease_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}")
        
    cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(raw_path, header=None, names=cols, na_values='?')
    
    # Preprocess missing values
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0] if not df['ca'].mode().empty else 0.0)
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0] if not df['thal'].mode().empty else 3.0)
    
    # Bounds check
    df = validate_and_clean_column(df, 'trestbps', 70, 220)
    df = validate_and_clean_column(df, 'chol', 100, 600)
    df = validate_and_clean_column(df, 'thalach', 60, 220)
    
    df = impute_missing_values(df)
    df = handle_outliers_iqr(df, exclude_cols=['target', 'sex', 'fbs', 'exang', 'restecg', 'cp', 'slope'])
    
    # Binarize outcome
    df['target'] = (df['target'] > 0).astype(int)
    
    out_path = os.path.join(PROCESSED_DIR, "heart_disease_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"Processed Heart Disease saved to {out_path}")

def preprocess_diabetes():
    raw_path = os.path.join(RAW_DIR, "diabetes_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}")
        
    cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df = pd.read_csv(raw_path, header=None, names=cols)
    
    # Physiological zeroes handling
    invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for field in invalid_zeros:
        # Glucose, BP, BMI can't be 0
        df[field] = df[field].replace(0, np.nan)
        
    df = validate_and_clean_column(df, 'Glucose', 40, 400)
    df = validate_and_clean_column(df, 'BloodPressure', 30, 200)
    df = validate_and_clean_column(df, 'BMI', 12, 75)
    
    df = impute_missing_values(df)
    df = handle_outliers_iqr(df, exclude_cols=['Outcome'])
    
    out_path = os.path.join(PROCESSED_DIR, "diabetes_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"Processed Diabetes saved to {out_path}")

def preprocess_breast_cancer():
    raw_path = os.path.join(RAW_DIR, "breast_cancer_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    # Find the same top features or read what sklearn loads
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Keep the structure robust, let's select the same top 10 features used in previous pipeline
    # ['worst area', 'worst concave points', 'mean concave points', 'worst radius', 'worst perimeter', 'mean perimeter', 'mean concavity', 'mean area', 'worst concavity', 'mean radius']
    top_10 = ['worst area', 'worst concave points', 'mean concave points', 'worst radius', 'worst perimeter', 
              'mean perimeter', 'mean concavity', 'mean area', 'worst concavity', 'mean radius']
    
    df_pruned = df[top_10 + ['target']].copy()
    df_pruned = impute_missing_values(df_pruned)
    df_pruned = handle_outliers_iqr(df_pruned, exclude_cols=['target'])
    
    out_path = os.path.join(PROCESSED_DIR, "breast_cancer_processed.csv")
    df_pruned.to_csv(out_path, index=False)
    print(f"Processed Breast Cancer saved to {out_path}")

def preprocess_ckd():
    raw_path = os.path.join(RAW_DIR, "chronic_kidney_disease_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    # Blood pressure and serum creatinine bounds
    df = validate_and_clean_column(df, 'bp', 40, 200)
    df = validate_and_clean_column(df, 'sc', 0.2, 30.0)
    df = validate_and_clean_column(df, 'hemo', 2.0, 20.0)
    
    df = impute_missing_values(df)
    df = handle_outliers_iqr(df, exclude_cols=['classification', 'htn'])
    
    out_path = os.path.join(PROCESSED_DIR, "chronic_kidney_disease_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"Processed Chronic Kidney Disease saved to {out_path}")

def preprocess_liver():
    raw_path = os.path.join(RAW_DIR, "liver_disease_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}")
        
    # Check if header exists by examining the first line
    with open(raw_path, 'r') as f:
        first_line = f.readline()
        
    cols = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A_G_Ratio', 'Outcome']
    if 'Age' in first_line or 'Gender' in first_line:
        df = pd.read_csv(raw_path)
    else:
        df = pd.read_csv(raw_path, header=None, names=cols)
        
    # Map Gender to binary numbers
    if df['Gender'].dtype == object:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, '1': 1, '0': 0, 1: 1, 0: 0})
    df['Gender'] = df['Gender'].fillna(1).astype(int)
    
    # Map Outcome to binary (1=disease, 0=no disease)
    # UCI liver dataset outcome is 1 (disease) or 2 (no disease)
    df['Outcome'] = df['Outcome'].map({1: 1, 2: 0, '1': 1, '2': 0, 0: 0})
    df['Outcome'] = df['Outcome'].fillna(0).astype(int)
    
    # Handle missing/invalid variables
    df = validate_and_clean_column(df, 'TB', 0.1, 100.0)
    df = validate_and_clean_column(df, 'Sgpt', 5, 5000)
    df = validate_and_clean_column(df, 'Sgot', 5, 5000)
    
    # Calculate A_G_Ratio where empty
    null_ratio = df['A_G_Ratio'].isnull()
    if null_ratio.any():
        df.loc[null_ratio, 'A_G_Ratio'] = df.loc[null_ratio, 'ALB'] / (df.loc[null_ratio, 'TP'] - df.loc[null_ratio, 'ALB'] + 1e-5)
        
    df = impute_missing_values(df)
    df = handle_outliers_iqr(df, exclude_cols=['Outcome', 'Gender'])
    
    out_path = os.path.join(PROCESSED_DIR, "liver_disease_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"Processed Liver Disease saved to {out_path}")

def preprocess_hypertension():
    raw_path = os.path.join(RAW_DIR, "hypertension_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    df = validate_and_clean_column(df, 'SystolicBP', 60, 260)
    df = validate_and_clean_column(df, 'DiastolicBP', 30, 160)
    df = validate_and_clean_column(df, 'Cholesterol', 80, 500)
    
    df = impute_missing_values(df)
    df = handle_outliers_iqr(df, exclude_cols=['Outcome', 'Sex', 'Smoking', 'FamilyHistory'])
    
    out_path = os.path.join(PROCESSED_DIR, "hypertension_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"Processed Hypertension saved to {out_path}")

def main():
    print("Executing automated preprocessing pipeline...")
    preprocess_heart_disease()
    preprocess_diabetes()
    preprocess_breast_cancer()
    preprocess_ckd()
    preprocess_liver()
    preprocess_hypertension()
    print("Preprocessing completed successfully.")

if __name__ == '__main__':
    main()
