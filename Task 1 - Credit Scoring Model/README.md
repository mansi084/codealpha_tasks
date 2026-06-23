# Credit Scoring and Creditworthiness Prediction System

An end-to-end Machine Learning system that predicts individual creditworthiness (Good/Bad Credit Risk) based on historical financial and customer behavior data. The platform features an automated preprocessing pipeline, custom feature engineering, hyperparameter-tuned classification models, explainable AI explanations (SHAP), a FastAPI backend service, a database version/history layer, and a Streamlit decisioning dashboard.

---

## Project Architecture & Directory Structure
```
credit_scoring_system/
│
├── config.yaml            # Central configuration file for features, hyperparams, and paths
├── requirements.txt       # System python dependencies
├── run_system.py          # Unified CLI entrypoint to manage the platform
│
├── data/
│   ├── generate_dataset.py # Synthetic financial data generator
│   └── credit_data.csv    # Raw training dataset (2500+ records)
│
├── database/
│   ├── db.py              # SQLAlchemy DB setup & session manager
│   ├── models.py          # SQLAlchemy tables (Applicants, Predictions, Versions, Metrics, Logs)
│   └── credit_scoring.db  # Active SQLite database (auto-generated)
│
├── preprocessing/
│   ├── data_validator.py  # Input validation schema and boundaries
│   └── preprocessor.py    # Standard scaling, missing value imputation, outlier winsorization
│
├── feature_engineering/
│   └── engineer.py        # Custom credit scoring features & feature importance graphs
│
├── training/
│   ├── train.py           # GridSearchCV comparison model trainer
│   └── retrain.py         # Automated retraining script
│
├── evaluation/
│   └── evaluator.py       # Metrics evaluator & curve plotting (ROC, Precision-Recall)
│
├── explainability/
│   └── explainer.py       # SHAP explainability calculations and local contribution graphs
│
├── backend/
│   ├── api.py             # FastAPI backend app & endpoints
│   ├── schemas/
│   │   └── applicant.py   # Pydantic validation schemas
│   └── services/
│       └── prediction.py  # Prediction mapping (prob to score), DB logging, & batch CSV processing
│
├── frontend/
│   └── app.py             # Streamlit premium dashboard
│
└── tests/
    ├── test_preprocessing.py # Preprocessing & validation tests
    ├── test_database.py      # SQLAlchemy query integrity tests
    └── test_api.py           # FastAPI endpoints integration tests
```

---

## Financial Dataset & Engineered Features

### Raw Customer Features:
* **Annual Income / Monthly Income**: Raw financials.
* **Loan Amount**: Desired credit limit request.
* **Existing Debts**: Current liabilities.
* **Debt-to-Income Ratio**: Debt vs income.
* **Number of Credit Cards / Credit Utilization Ratio**: Card activity metrics.
* **Payment History**: Qualitative metric (`Excellent`, `Good`, `Fair`, `Poor`).
* **Number of Late Payments**: Count of delayed payments.
* **Loan Repayment History**: Historical behavior (`All Paid`, `Mostly Paid`, `Delayed`, `Defaulted`).
* **Employment Length / Age / Savings Balance / Previous Defaults / Credit History Length**.

### Engineered Derived Features:
1. **Refined Debt-to-Income Ratio**: Refined ratio combining annual income and existing debts.
2. **Credit Utilization Percentage**: Credit card utilization expressed as a percentage.
3. **Payment Consistency Score**: Composite metric penalizing late payments based on categorical history rating.
4. **Financial Stability Score**: Scaled value indicating financial robustness (savings, employment length, age vs debts).
5. **Risk Index**: Sum of weighted risk markers (late payments, defaults, high utilization, and loan ratio).
6. **Credit Age Score**: Proportion of credit history relative to mature adult age.

---

## SQLite Database Schema
The database tracks predictions, model health metrics, versions, and user logs:
* **`applicants`**: Stores basic info, age, income, and financial features.
* **`prediction_history`**: Links to applicants, stores the credit score, probability, risk category, recommendation, and active model version.
* **`model_versions`**: Records trained models, pickle file paths, and deployment statuses.
* **`evaluation_metrics`**: Records accuracy, precision, recall, F1, and ROC-AUC for each model version.
* **`user_activity_logs`**: Logs model training runs, API requests, and retraining pipelines.

---

## FastAPI Backend Endpoints
* **`POST /predict`**: Accepts single applicant financial details. Evaluates the ML model, maps probability to a credit score range of 300-850, saves the customer records, and outputs predictions.
* **`GET /model-metrics`**: Returns metrics for all registered historical model versions.
* **`GET /feature-importance`**: Returns relative feature importance of the active model.
* **`POST /batch-predict`**: Process multi-record predictions from an uploaded CSV file.
* **`GET /predict/{applicant_id}/shap-plot`**: Generates and streams a custom local SHAP explanation plot.

---

## Streamlit Frontend Dashboard
Includes:
1. **Real-time Assessment**: Multi-column applicant form with sliders, numeric inputs, and automatic DTI calculation. Displays credit score progress bars, risk metrics, and SHAP explainability breakdown plots.
2. **Batch Prediction**: Offers CSV template downloads, batch file uploads, summary indicators (Approval/Review/Denial rates), and result file downloads.
3. **Model Performance**: Visualizes accuracy, recall, and historical metrics in tabular format alongside ROC, Precision-Recall, and Confusion Matrix charts.
4. **Global Feature Importance**: Interactive plots and datasets showing overall model priorities.

---

## Quick Start & Deployment Guide

### 1. Installation
Clone the workspace and install python packages:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
To fit the preprocessor, compare the 5 classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost, and Gradient Boosting) via GridSearchCV, select the best model, and generate evaluation plots:
```bash
python run_system.py train
```

### 3. Running Unit Tests
Validate that preprocessing, SQL mapping, and API routes work correctly:
```bash
python run_system.py test
```

### 4. Running the System (API + Dashboard)
To run both the FastAPI server (port 8000) and the Streamlit dashboard (port 8501) simultaneously:
```bash
python run_system.py all
```
* Access the Swagger API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)
* Access the Streamlit Dashboard at: [http://localhost:8501](http://localhost:8501)

### 5. Automated Retraining
Trigger the automated retraining script to ingest new logs/records, regenerate datasets, and dynamically activate the best model:
```bash
python run_system.py retrain
```
