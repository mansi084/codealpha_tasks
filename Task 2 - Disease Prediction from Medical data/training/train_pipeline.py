import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from database.connection import SessionLocal
from database.models import ModelVersion

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TrainPipeline")

DATA_DIR = "data"
MODEL_DIR = "models"
STATIC_DIR = "static"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

def generate_subsampled_curve(y_test, y_probs, curve_type='roc', samples=20):
    """Subsamples ROC or PR curve points to keep JSON payloads lightweight."""
    if curve_type == 'roc':
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        indices = np.linspace(0, len(fpr) - 1, num=samples, dtype=int)
        pts = [{'fpr': float(fpr[i]), 'tpr': float(tpr[i])} for i in indices]
        if pts[-1]['fpr'] != 1.0 or pts[-1]['tpr'] != 1.0:
            pts.append({'fpr': 1.0, 'tpr': 1.0})
        return pts
    else: # pr curve
        prec, rec, _ = precision_recall_curve(y_test, y_probs)
        indices = np.linspace(0, len(prec) - 1, num=samples, dtype=int)
        pts = [{'recall': float(rec[i]), 'precision': float(prec[i])} for i in indices]
        return pts

def train_and_evaluate_disease(disease_key, df, target_col, display_name, feature_labels):
    logger.info(f"--- Training pipeline started for: {display_name} ({disease_key}) ---")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    features = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_file = os.path.join(MODEL_DIR, f"{disease_key}_scaler.joblib")
    joblib.dump(scaler, scaler_file)
    logger.info(f"Saved scaler to {scaler_file}")
    
    # Define models to train and hyperparameter grids
    models_config = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'grid': {'C': [0.1, 1.0, 10.0]}
        },
        'svm': {
            'model': SVC(probability=True, random_state=42),
            'grid': {'C': [0.5, 1.0, 5.0], 'kernel': ['rbf', 'linear']}
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'grid': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
        },
        'xgboost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=42),
            'grid': {'n_estimators': [50, 100], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'grid': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
        }
    }
    
    trained_models = {}
    metrics_log = {}
    best_algo = None
    best_auc = -1.0
    
    # Train and tune individual models
    for name, config in models_config.items():
        logger.info(f"Grid searching hyperparameter tuning for {name}...")
        grid_search = GridSearchCV(
            config['model'], config['grid'], cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # Fit model on training set
        best_model.fit(X_train_scaled, y_train)
        
        trained_models[name] = best_model
        model_path = os.path.join(MODEL_DIR, f"{disease_key}_{name}.joblib")
        joblib.dump(best_model, model_path)
        
        # Predict
        y_pred = best_model.predict(X_test_scaled)
        y_probs = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_probs)
        
        # Sensitivity / Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
        
        roc_pts = generate_subsampled_curve(y_test, y_probs, 'roc')
        pr_pts = generate_subsampled_curve(y_test, y_probs, 'pr')
        
        # Global Feature Importance
        feature_weights = {}
        if hasattr(best_model, 'feature_importances_'):
            for f, w in zip(features, best_model.feature_importances_):
                feature_weights[f] = float(w)
        elif hasattr(best_model, 'coef_'):
            for f, w in zip(features, best_model.coef_[0]):
                feature_weights[f] = float(abs(w)) # Use magnitude of weights
        else:
            feature_weights = {f: 0.0 for f in features}
            
        metrics_log[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            },
            'roc_curve': roc_pts,
            'pr_curve': pr_pts,
            'feature_importance': feature_weights
        }
        
        logger.info(f"[{name}] Test Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_algo = name
            
    # Implement Ensemble Learning (Voting Classifier of best Estimators)
    logger.info("Training Ensemble Voting Classifier...")
    ensemble_estimators = []
    # Use best tuned Logistic Regression, Random Forest, and XGBoost
    for name in ['logistic_regression', 'random_forest', 'xgboost']:
        ensemble_estimators.append((name, trained_models[name]))
        
    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    voting_clf.fit(X_train_scaled, y_train)
    
    # Save ensemble
    ensemble_path = os.path.join(MODEL_DIR, f"{disease_key}_ensemble.joblib")
    joblib.dump(voting_clf, ensemble_path)
    
    y_pred_ens = voting_clf.predict(X_test_scaled)
    y_probs_ens = voting_clf.predict_proba(X_test_scaled)[:, 1]
    
    acc_ens = accuracy_score(y_test, y_pred_ens)
    prec_ens = precision_score(y_test, y_pred_ens, zero_division=0)
    rec_ens = recall_score(y_test, y_pred_ens, zero_division=0)
    f1_ens = f1_score(y_test, y_pred_ens, zero_division=0)
    auc_ens = roc_auc_score(y_test, y_probs_ens)
    
    tn_ens, fp_ens, fn_ens, tp_ens = confusion_matrix(y_test, y_pred_ens).ravel()
    sens_ens = float(tp_ens) / (tp_ens + fn_ens) if (tp_ens + fn_ens) > 0 else 0.0
    spec_ens = float(tn_ens) / (tn_ens + fp_ens) if (tn_ens + fp_ens) > 0 else 0.0
    
    roc_pts_ens = generate_subsampled_curve(y_test, y_probs_ens, 'roc')
    pr_pts_ens = generate_subsampled_curve(y_test, y_probs_ens, 'pr')
    
    # Feature importance for ensemble is average of underlying models if applicable
    ensemble_importance = {}
    for f in features:
        rf_imp = metrics_log['random_forest']['feature_importance'].get(f, 0)
        xgb_imp = metrics_log['xgboost']['feature_importance'].get(f, 0)
        lr_imp = metrics_log['logistic_regression']['feature_importance'].get(f, 0)
        ensemble_importance[f] = float((rf_imp + xgb_imp + lr_imp) / 3.0)
        
    metrics_log['ensemble'] = {
        'accuracy': float(acc_ens),
        'precision': float(prec_ens),
        'recall': float(rec_ens),
        'f1_score': float(f1_ens),
        'roc_auc': float(auc_ens),
        'sensitivity': sens_ens,
        'specificity': spec_ens,
        'confusion_matrix': {
            'tn': int(tn_ens), 'fp': int(fp_ens), 'fn': int(fn_ens), 'tp': int(tp_ens)
        },
        'roc_curve': roc_pts_ens,
        'pr_curve': pr_pts_ens,
        'feature_importance': ensemble_importance
    }
    logger.info(f"[ensemble] Test Accuracy: {acc_ens:.4f}, ROC-AUC: {auc_ens:.4f}")
    
    if auc_ens > best_auc:
        best_auc = auc_ens
        best_algo = 'ensemble'
        
    # Save the absolute best model
    best_model_obj = voting_clf if best_algo == 'ensemble' else trained_models[best_algo]
    best_model_path = os.path.join(MODEL_DIR, f"{disease_key}_best.joblib")
    joblib.dump(best_model_obj, best_model_path)
    logger.info(f"Best model for {disease_key} is {best_algo} with ROC-AUC {best_auc:.4f}. Saved to {best_model_path}")
    
    # Save metadata to DB
    try:
        db = SessionLocal()
        # Check if record already exists
        existing = db.query(ModelVersion).filter_by(disease_type=disease_key, version="1.0.0").first()
        if existing:
            db.delete(existing)
            
        model_ver = ModelVersion(
            disease_type=disease_key,
            version="1.0.0",
            algorithm=best_algo,
            accuracy=metrics_log[best_algo]['accuracy'],
            f1_score=metrics_log[best_algo]['f1_score'],
            roc_auc=metrics_log[best_algo]['roc_auc'],
            file_path=best_model_path
        )
        db.add(model_ver)
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"Error logging model details to database: {e}")
        
    return {
        'display_name': display_name,
        'features': features,
        'feature_labels': feature_labels,
        'best_algorithm': best_algo,
        'models': metrics_log
    }

def main():
    logger.info("Executing global training and evaluation pipeline...")
    
    datasets_info = {
        'heart_disease': {
            'file': 'heart_disease_processed.csv',
            'target': 'target',
            'display_name': 'Heart Disease Prediction',
            'labels': {
                'age': 'Age (years)', 'sex': 'Sex (1=Male, 0=Female)', 'cp': 'Chest Pain Type (1-4)',
                'trestbps': 'Resting BP (mmHg)', 'chol': 'Cholesterol (mg/dL)', 'fbs': 'Fasting Blood Sugar (1=Yes, 0=No)',
                'restecg': 'Resting ECG (0-2)', 'thalach': 'Max Heart Rate (bpm)', 'exang': 'Exercise Angina (1=Yes, 0=No)',
                'oldpeak': 'ST Depression', 'slope': 'Slope of ST Segment (1-3)', 'ca': 'Major Vessels Colored (0-3)',
                'thal': 'Thalassemia Result (3=Normal, 6=Fixed, 7=Reversible)'
            }
        },
        'diabetes': {
            'file': 'diabetes_processed.csv',
            'target': 'Outcome',
            'display_name': 'Diabetes Risk Prediction',
            'labels': {
                'Pregnancies': 'Pregnancies Count', 'Glucose': 'Glucose Level (mg/dL)',
                'BloodPressure': 'Blood Pressure (mmHg)', 'SkinThickness': 'Skinfold Thickness (mm)',
                'Insulin': 'Insulin Level (μU/mL)', 'BMI': 'BMI (kg/m²)',
                'DiabetesPedigreeFunction': 'Diabetes Pedigree Score', 'Age': 'Age (years)'
            }
        },
        'breast_cancer': {
            'file': 'breast_cancer_processed.csv',
            'target': 'target',
            'display_name': 'Breast Cancer Classification',
            'labels': {
                'worst area': 'Worst Area', 'worst concave points': 'Worst Concave Points',
                'mean concave points': 'Mean Concave Points', 'worst radius': 'Worst Radius',
                'worst perimeter': 'Worst Perimeter', 'mean perimeter': 'Mean Perimeter',
                'mean concavity': 'Mean Concavity', 'mean area': 'Mean Area',
                'worst concavity': 'Worst Concavity', 'mean radius': 'Mean Radius'
            }
        },
        'chronic_kidney_disease': {
            'file': 'chronic_kidney_disease_processed.csv',
            'target': 'classification',
            'display_name': 'Chronic Kidney Disease Risk',
            'labels': {
                'age': 'Age (years)', 'bp': 'Blood Pressure (mmHg)', 'sg': 'Specific Gravity',
                'al': 'Albumin Level (0-4)', 'bgr': 'Blood Glucose Random (mg/dL)', 'bu': 'Blood Urea (mg/dL)',
                'sc': 'Serum Creatinine (mg/dL)', 'hemo': 'Hemoglobin (g/dL)', 'pcv': 'Packed Cell Volume (%)',
                'htn': 'Hypertension (1=Yes, 0=No)'
            }
        },
        'liver_disease': {
            'file': 'liver_disease_processed.csv',
            'target': 'Outcome',
            'display_name': 'Liver Disease Prediction',
            'labels': {
                'Age': 'Age (years)', 'Gender': 'Gender (1=Male, 0=Female)', 'TB': 'Total Bilirubin (mg/dL)',
                'DB': 'Direct Bilirubin (mg/dL)', 'Alkphos': 'Alkaline Phosphatase (U/L)',
                'Sgpt': 'SGPT (ALT) Level (U/L)', 'Sgot': 'SGOT (AST) Level (U/L)',
                'TP': 'Total Proteins (g/dL)', 'ALB': 'Albumin (g/dL)', 'A_G_Ratio': 'A/G Ratio'
            }
        },
        'hypertension': {
            'file': 'hypertension_processed.csv',
            'target': 'Outcome',
            'display_name': 'Hypertension Risk Assessment',
            'labels': {
                'Age': 'Age (years)', 'Sex': 'Sex (1=Male, 0=Female)', 'BMI': 'BMI (kg/m²)',
                'SystolicBP': 'Systolic BP (mmHg)', 'DiastolicBP': 'Diastolic BP (mmHg)',
                'Cholesterol': 'Cholesterol (mg/dL)', 'HeartRate': 'Heart Rate (bpm)',
                'Smoking': 'Smoking Status (1=Yes, 0=No)', 'FamilyHistory': 'Family History (1=Yes, 0=No)'
            }
        }
    }
    
    global_metrics = {}
    
    for key, info in datasets_info.items():
        file_path = os.path.join(DATA_DIR, info['file'])
        if not os.path.exists(file_path):
            logger.error(f"Processed dataset not found: {file_path}. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        disease_results = train_and_evaluate_disease(
            key, df, info['target'], info['display_name'], info['labels']
        )
        global_metrics[key] = disease_results
        
    # Save the output global JSON metric file
    metrics_path = os.path.join(STATIC_DIR, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(global_metrics, f, indent=4)
        
    logger.info(f"Training pipeline complete. Model evaluation stats written to: {metrics_path}")

if __name__ == '__main__':
    main()
