import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer derived features from raw inputs.
    Works on both preprocessed and raw DataFrames.
    """
    df_out = df.copy()
    
    # Fill temp NAs if not preprocessed to avoid math errors during feature extraction
    annual_income = df_out['annual_income'].fillna(df_out['annual_income'].median() if 'annual_income' in df_out else 50000)
    existing_debts = df_out['existing_debts'].fillna(df_out['existing_debts'].median() if 'existing_debts' in df_out else 10000)
    savings_balance = df_out['savings_balance'].fillna(df_out['savings_balance'].median() if 'savings_balance' in df_out else 5000)
    loan_amount = df_out['loan_amount'].fillna(df_out['loan_amount'].median() if 'loan_amount' in df_out else 20000)
    credit_util = df_out['credit_utilization_ratio'].fillna(df_out['credit_utilization_ratio'].median() if 'credit_utilization_ratio' in df_out else 0.3)
    late_payments = df_out['number_of_late_payments'].fillna(df_out['number_of_late_payments'].median() if 'number_of_late_payments' in df_out else 0)
    emp_len = df_out['employment_length'].fillna(df_out['employment_length'].median() if 'employment_length' in df_out else 3)
    age = df_out['age'].fillna(df_out['age'].median() if 'age' in df_out else 35)
    credit_hist = df_out['credit_history_length'].fillna(df_out['credit_history_length'].median() if 'credit_history_length' in df_out else 5)
    
    # Explicit conversion mappings for categorical values for manual calculations (if categories are strings)
    pay_history_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    prev_defaults_map = {'No': 0, 'Yes': 1}
    
    if 'payment_history' in df_out.columns:
        if df_out['payment_history'].dtype == object or isinstance(df_out['payment_history'].iloc[0], str):
            pay_hist_enc = df_out['payment_history'].map(pay_history_map).fillna(1).values
        else:
            pay_hist_enc = df_out['payment_history'].values
    else:
        pay_hist_enc = np.ones(len(df_out)) * 2
        
    if 'previous_defaults' in df_out.columns:
        if df_out['previous_defaults'].dtype == object or isinstance(df_out['previous_defaults'].iloc[0], str):
            prev_def_enc = df_out['previous_defaults'].map(prev_defaults_map).fillna(0).values
        else:
            prev_def_enc = df_out['previous_defaults'].values
    else:
        prev_def_enc = np.zeros(len(df_out))

    # 1. Debt-to-Income Ratio Refined
    df_out['debt_to_income_ratio_refined'] = np.round(existing_debts / (annual_income + 1.0), 4)
    
    # 2. Credit Utilization Percentage
    df_out['credit_utilization_percentage'] = np.round(credit_util * 100.0, 2)
    
    # 3. Payment Consistency Score
    # Range: Higher is better consistency. E.g. excellent history with 0 late payments = 3.0.
    df_out['payment_consistency_score'] = np.round(pay_hist_enc - (late_payments * 0.25), 4)
    
    # 4. Financial Stability Score
    # Scale positive indicators (savings, employment length, age stability) over debt ratio.
    stability = (np.log1p(savings_balance) * (emp_len + 1.0)) / (df_out['debt_to_income_ratio_refined'] + 0.15)
    df_out['financial_stability_score'] = np.round(stability, 4)
    
    # 5. Risk Index
    # Scale negative indicators. Higher means more risky.
    risk = (late_payments * 1.5) + (prev_def_enc * 4.0) + (credit_util * 3.0) + (loan_amount / (annual_income + 1e-5) * 2.0)
    df_out['risk_index'] = np.round(risk, 4)
    
    # 6. Credit Age Score
    # How long credit has been open relative to adult age (age - 18)
    age_denominator = np.clip(age - 17.0, 1.0, None)
    df_out['credit_age_score'] = np.round(credit_hist / age_denominator, 4)
    
    return df_out

def plot_save_feature_importance(importances, feature_names, save_path):
    """
    Generate and save feature importance graph.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Plotting using Seaborn
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title("Feature Importance Analysis", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Feature importance graph saved to {save_path}")
