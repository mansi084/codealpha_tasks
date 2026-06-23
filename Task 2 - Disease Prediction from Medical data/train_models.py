import os
import json
import logging
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.datasets import load_breast_cancer

# Config logging framework for production-grade output
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MLPipeline")

DATA_DIR = "data"
MODEL_DIR = "models"
STATIC_DIR = "static"

# Clinical datasets download points
RESOURCES = {
    "heart": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "diabetes": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
}

def setup_workspace():
    """Build project directory tree."""
    for folder in [DATA_DIR, MODEL_DIR, STATIC_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f"Initialized directory: {folder}")

def load_heart_dataset():
    """
    Downloads, parses, and cleans the Cleveland heart disease dataset.
    Binarizes target clinical scores to binary risk levels.
    """
    csv_file = os.path.join(DATA_DIR, "heart_disease_raw.csv")
    
    if not os.path.exists(csv_file):
        logger.info("Local heart dataset cache miss. Fetching from UCI Repository...")
        try:
            urllib.request.urlretrieve(RESOURCES["heart"], csv_file)
        except Exception as err:
            logger.warning(f"Could not retrieve heart data: {err}. Building local synthetic fallback.")
            # Fallback mock engine generates statistical approximations of Cleveland parameters
            np.random.seed(101)
            n = 300
            mock_data = {
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
            }
            pd.DataFrame(mock_data).to_csv(csv_file, index=False, header=False)

    cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    df = pd.read_csv(csv_file, header=None, names=cols, na_values='?')
    
    # Resolve missing values (marked as '?' in the raw data) using standard mode replacement
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    
    # Convert multi-class diagnosis target to binary diagnostic classification label
    df['target'] = (df['target'] > 0).astype(int)
    
    df.to_csv(os.path.join(DATA_DIR, "heart_disease_processed.csv"), index=False)
    return df

def load_diabetes_dataset():
    """Loads and sanitizes Pima Indians Diabetes Dataset by resolving physiological zeros."""
    csv_file = os.path.join(DATA_DIR, "diabetes_raw.csv")
    
    if not os.path.exists(csv_file):
        logger.info("Local diabetes dataset cache miss. Fetching from remote source...")
        try:
            urllib.request.urlretrieve(RESOURCES["diabetes"], csv_file)
        except Exception as err:
            logger.warning(f"Could not retrieve diabetes data: {err}. Building local synthetic fallback.")
            np.random.seed(202)
            n = 768
            mock_data = {
                'Pregnancies': np.random.poisson(3.8, n),
                'Glucose': np.random.normal(121.0, 31.0, n).astype(int),
                'BloodPressure': np.random.normal(69.0, 19.0, n).astype(int),
                'SkinThickness': np.random.normal(20.5, 15.5, n).astype(int),
                'Insulin': np.random.normal(80.0, 115.0, n).astype(int),
                'BMI': np.random.normal(32.0, 7.5, n),
                'DiabetesPedigreeFunction': np.random.exponential(0.47, n),
                'Age': np.random.normal(33.0, 11.5, n).astype(int),
                'Outcome': np.random.choice([0, 1], n, p=[0.65, 0.35])
            }
            for key in mock_data:
                if key != 'Outcome':
                    mock_data[key] = np.clip(mock_data[key], 0, None)
            pd.DataFrame(mock_data).to_csv(csv_file, index=False, header=False)

    cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df = pd.read_csv(csv_file, header=None, names=cols)
    
    # Values of 0 are biologically invalid for these columns; replace with median estimates
    invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for field in invalid_zeros:
        df[field] = df[field].replace(0, np.nan)
        df[field] = df[field].fillna(df[field].median())
        
    df.to_csv(os.path.join(DATA_DIR, "diabetes_processed.csv"), index=False)
    return df

def load_cancer_dataset():
    """Loads Wisconsin Breast Cancer dataset, extracting top 10 features via Random Forest."""
    logger.info("Loading sklearn built-in Breast Cancer dataset...")
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Feature selector using ensemble importance scoring
    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(X, y)
    
    ranks = np.argsort(selector.feature_importances_)[::-1]
    top_10 = [X.columns[i] for i in ranks[:10]]
    
    logger.info(f"Selected Top 10 predictive features: {top_10}")
    
    df_pruned = df[top_10 + ['target']]
    df_pruned.to_csv(os.path.join(DATA_DIR, "breast_cancer_processed.csv"), index=False)
    
    return df_pruned, top_10

def generate_subsampled_roc(y_test, y_probs, samples=25):
    """Generates subsampled points from the ROC curve to keep output payload clean."""
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    indices = np.linspace(0, len(fpr) - 1, num=samples, dtype=int)
    
    pts = [{'fpr': float(fpr[i]), 'tpr': float(tpr[i])} for i in indices]
    
    # Enforce terminal bounds
    if pts[-1]['fpr'] != 1.0 or pts[-1]['tpr'] != 1.0:
        pts.append({'fpr': 1.0, 'tpr': 1.0})
    return pts

def execute_pipeline():
    setup_workspace()
    
    datasets = {
        'heart_disease': {
            'df': load_heart_dataset(),
            'target': 'target',
            'display_name': 'Heart Disease Prediction',
            'labels': {
                'age': 'Age (years)',
                'sex': 'Sex (1=Male, 0=Female)',
                'cp': 'Chest Pain Type (1-4)',
                'trestbps': 'Resting Blood Pressure (mmHg)',
                'chol': 'Cholesterol (mg/dl)',
                'fbs': 'Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)',
                'restecg': 'Resting ECG Results (0-2)',
                'thalach': 'Max Heart Rate Achieved (bpm)',
                'exang': 'Exercise Induced Angina (1=Yes, 0=No)',
                'oldpeak': 'ST Depression (oldpeak)',
                'slope': 'Slope of ST Segment (1-3)',
                'ca': 'Major Vessels Colored (0-3)',
                'thal': 'Thalassemia (3=Normal, 6=Fixed, 7=Reversible)'
            }
        },
        'diabetes': {
            'df': load_diabetes_dataset(),
            'target': 'Outcome',
            'display_name': 'Diabetes Risk Prediction',
            'labels': {
                'Pregnancies': 'Pregnancies Count',
                'Glucose': 'Glucose Level (mg/dL)',
                'BloodPressure': 'Blood Pressure (mmHg)',
                'SkinThickness': 'Skinfold Thickness (mm)',
                'Insulin': 'Insulin Level (μU/mL)',
                'BMI': 'BMI (kg/m²)',
                'DiabetesPedigreeFunction': 'Diabetes Pedigree Score',
                'Age': 'Age (years)'
            }
        }
    }
    
    bc_df, bc_features = load_cancer_dataset()
    datasets['breast_cancer'] = {
        'df': bc_df,
        'target': 'target',
        'display_name': 'Breast Cancer Classification',
        'labels': {feat: feat.replace('mean ', '').title() for feat in bc_features}
    }
    
    pipeline_metrics = {}
    
    for key, info in datasets.items():
        logger.info(f"Training models for target condition: {info['display_name']}")
        df = info['df']
        target = info['target']
        
        X = df.drop(columns=[target])
        y = df[target]
        features = list(X.columns)
        
        # Consistent stratified splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        scaler_file = os.path.join(MODEL_DIR, f"{key}_scaler.joblib")
        joblib.dump(scaler, scaler_file)
        
        classifiers = {
            'logistic_regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
            'svm': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'xgboost': XGBClassifier(eval_metric='logloss', max_depth=5, random_state=42)
        }
        
        pipeline_metrics[key] = {
            'display_name': info['display_name'],
            'features': features,
            'feature_labels': info['labels'],
            'models': {}
        }
        
        for name, clf in classifiers.items():
            clf.fit(X_train_scaled, y_train)
            
            y_pred = clf.predict(X_test_scaled)
            y_probs = clf.predict_proba(X_test_scaled)[:, 1]
            
            model_file = os.path.join(MODEL_DIR, f"{key}_{name}.joblib")
            joblib.dump(clf, model_file)
            
            # Compute evaluation parameters
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_probs)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            roc_points = generate_subsampled_roc(y_test, y_probs, samples=20)
            
            # Extract importances
            feature_weights = {}
            if hasattr(clf, 'feature_importances_'):
                for f, w in zip(features, clf.feature_importances_):
                    feature_weights[f] = float(w)
            elif hasattr(clf, 'coef_'):
                for f, w in zip(features, clf.coef_[0]):
                    feature_weights[f] = float(w)
            else:
                feature_weights = {f: 0.0 for f in features}
                
            pipeline_metrics[key]['models'][name] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(auc),
                'confusion_matrix': {
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                },
                'roc_curve': roc_points,
                'feature_importance': feature_weights
            }
            logger.info(f"[{key} -> {name}] Evaluated: Accuracy={acc:.4f}, AUC={auc:.4f}")

    metrics_file = os.path.join(STATIC_DIR, "model_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(pipeline_metrics, f, indent=4)
        
    logger.info(f"Pipeline complete. Evaluation parameters exported to {metrics_file}")

if __name__ == '__main__':
    execute_pipeline()
