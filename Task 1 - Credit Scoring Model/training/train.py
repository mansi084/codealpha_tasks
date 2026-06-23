import os
import json
import yaml
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Import local modules
from database.db import init_db, Session
from database.models import ModelVersion, EvaluationMetric, UserActivityLog
from preprocessing.data_validator import DataValidator
from preprocessing.preprocessor import Preprocessor
from feature_engineering.engineer import add_derived_features, plot_save_feature_importance
from evaluation.evaluator import evaluate_classifier, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_training")

def train_pipeline(data_path=None):
    # Initialize DB schema
    init_db()
    db = Session()
    
    # 1. Load Config
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if not data_path:
        data_path = config['paths']['raw_data']
        
    logger.info(f"Loading raw dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    # 2. Data Validation
    num_features = config['features']['numerical']
    cat_features = config['features']['categorical']
    target_col = config['features']['target']
    
    validator = DataValidator(num_features, cat_features, target_col)
    is_valid, validation_errors = validator.validate(df_raw, is_training=True)
    if not is_valid:
        logger.error(f"Data validation failed: {validation_errors}")
        db.add(UserActivityLog(action="Train Model - Failed", details=f"Data validation errors: {str(validation_errors)}"))
        db.commit()
        db.close()
        raise ValueError("Data validation failed. Cannot proceed with training.")
        
    # 3. Fit Preprocessor
    logger.info("Initializing and fitting preprocessor...")
    preprocessor = Preprocessor(num_features, cat_features, target_col)
    preprocessor.fit(df_raw)
    
    # Save preprocessor first
    preprocessor_path = config['paths']['preprocessor_path']
    preprocessor.save(preprocessor_path)
    
    # 4. Preprocess Data
    df_preprocessed = preprocessor.transform(df_raw)
    
    # 5. Feature Engineering
    logger.info("Engineering derived features...")
    df_engineered = add_derived_features(df_preprocessed)
    
    # Define features and target after preprocessing and feature engineering
    # Target values: Good=1, Bad=0
    y = df_engineered[target_col].astype(int)
    # Features exclude target
    X = df_engineered.drop(columns=[target_col])
    
    # Store features list for backend schemas/models
    feature_names = X.columns.tolist()
    logger.info(f"Final feature list ({len(feature_names)} features): {feature_names}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Define candidate classifiers and search space
    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'xgboost': XGBClassifier(random_state=42, use_label_encoder=False)
    }
    
    hparams = config['hyperparameters']
    
    # 7. Hyperparameter tuning & Comparison
    best_models = {}
    best_scores = {}
    evaluation_results = {}
    
    for name, clf in classifiers.items():
        logger.info(f"Running Grid Search for {name}...")
        grid_params = hparams.get(name, {})
        # Handle key discrepancies if any, or adjust grid search CV
        grid_search = GridSearchCV(estimator=clf, param_grid=grid_params, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_clf = grid_search.best_estimator_
        best_models[name] = best_clf
        best_scores[name] = grid_search.best_score_
        logger.info(f"Best parameters for {name}: {grid_search.best_params_} (CV ROC-AUC: {grid_search.best_score_:.4f})")
        
        # Predict on test set
        y_pred = best_clf.predict(X_test)
        y_prob = best_clf.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else y_pred
        
        # Evaluate
        metrics = evaluate_classifier(y_test, y_pred, y_prob)
        evaluation_results[name] = {
            'metrics': metrics,
            'model': best_clf
        }
        
    # 8. Identify the best model based on F1 score or ROC-AUC
    # We will use ROC-AUC as it's standard for credit scoring models (focusing on ranking probability)
    best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k]['metrics']['roc_auc'])
    best_run = evaluation_results[best_model_name]
    best_model = best_run['model']
    best_metrics = best_run['metrics']
    
    logger.info(f"===== BEST MODEL SELECTED: {best_model_name} with ROC-AUC {best_metrics['roc_auc']:.4f} =====")
    
    # 9. Save Best Model
    best_model_path = config['paths']['best_model_path']
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Best model saved to {best_model_path}")
    
    # 9b. Fit and save SHAP Explainer
    from explainability.explainer import CreditExplainer
    logger.info("Initializing SHAP Explainer on training data...")
    explainer = CreditExplainer(best_model, X_train)
    explainer_path = config['paths']['explainer_path']
    explainer.save(explainer_path)
    
    # 10. Generate and Save Curves/Plots for the Best Model
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    
    plots_dir = config['paths']['plots_dir']
    plot_confusion_matrix(y_test, y_test_pred, os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_test_prob, os.path.join(plots_dir, 'roc_curve.png'))
    plot_precision_recall_curve(y_test, y_test_prob, os.path.join(plots_dir, 'precision_recall_curve.png'))
    
    # 11. Feature Importance for Best Model (if supported)
    # Check if model has feature_importances_ or coef_
    importances = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        
    if importances is not None:
        plot_save_feature_importance(
            importances, 
            feature_names, 
            os.path.join(plots_dir, 'feature_importance.png')
        )
        # Save feature importance values in a JSON alongside metrics
        feature_importance_dict = dict(zip(feature_names, [float(v) for v in importances]))
    else:
        feature_importance_dict = {}
        
    # Save metrics JSON file
    metrics_path = config['paths']['metrics_path']
    summary_metrics = {
        'best_model': best_model_name,
        'features': feature_names,
        'metrics': best_metrics,
        'feature_importance': feature_importance_dict
    }
    with open(metrics_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
        
    # 12. Register model version in DB
    # Determine new version code (e.g. check current versions count)
    existing_versions_count = db.query(ModelVersion).count()
    version_str = f"v1.{existing_versions_count}.0"
    
    # Deactivate other model versions
    db.query(ModelVersion).update({ModelVersion.is_active: False})
    
    # Save the new version
    db_model_version = ModelVersion(
        version=version_str,
        model_name=best_model_name,
        file_path=best_model_path,
        is_active=True
    )
    db.add(db_model_version)
    db.commit() # Commit to get ID
    
    # Save metrics
    db_metrics = EvaluationMetric(
        model_version=version_str,
        accuracy=best_metrics['accuracy'],
        precision=best_metrics['precision'],
        recall=best_metrics['recall'],
        f1_score=best_metrics['f1_score'],
        roc_auc=best_metrics['roc_auc']
    )
    db.add(db_metrics)
    
    # Log activity
    activity = UserActivityLog(
        action="Train Model - Success",
        details=f"Trained {best_model_name} as version {version_str}. ROC-AUC: {best_metrics['roc_auc']:.4f}"
    )
    db.add(activity)
    db.commit()
    db.close()
    
    logger.info(f"Model training pipeline completed. Version registered: {version_str}")
    return version_str

if __name__ == "__main__":
    train_pipeline()
