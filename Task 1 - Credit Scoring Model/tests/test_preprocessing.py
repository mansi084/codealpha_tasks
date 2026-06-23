import pytest
import pandas as pd
import numpy as np
from preprocessing.data_validator import DataValidator
from preprocessing.preprocessor import Preprocessor
from feature_engineering.engineer import add_derived_features

def test_data_validator():
    num_cols = ['annual_income', 'age']
    cat_cols = ['payment_history']
    target_col = 'creditworthiness'
    
    validator = DataValidator(num_cols, cat_cols, target_col)
    
    # Valid data
    df_valid = pd.DataFrame({
        'annual_income': [50000.0, 60000.0],
        'age': [30, 45],
        'payment_history': ['Good', 'Excellent'],
        'creditworthiness': ['Good', 'Bad']
    })
    is_valid, errors = validator.validate(df_valid, is_training=True)
    assert is_valid is True
    assert len(errors) == 0
    
    # Invalid data (negative age, negative income, wrong categorical)
    df_invalid = pd.DataFrame({
        'annual_income': [-50000.0, 60000.0],
        'age': [12, 45],
        'payment_history': ['WrongCategory', 'Excellent'],
        'creditworthiness': ['Good', 'Bad']
    })
    is_valid, errors = validator.validate(df_invalid, is_training=True)
    assert is_valid is False
    assert len(errors) > 0

def test_preprocessor_imputation_and_scaling():
    num_cols = ['annual_income', 'age']
    cat_cols = ['payment_history']
    target_col = 'creditworthiness'
    
    preprocessor = Preprocessor(num_cols, cat_cols, target_col)
    
    # Training data with some NaNs
    df_train = pd.DataFrame({
        'annual_income': [50000.0, 60000.0, np.nan, 80000.0],
        'age': [30, np.nan, 40, 50],
        'payment_history': ['Good', 'Excellent', 'Fair', np.nan],
        'creditworthiness': ['Good', 'Bad', 'Good', 'Bad']
    })
    
    preprocessor.fit(df_train)
    assert preprocessor.is_fitted is True
    
    # Check that medians were captured
    assert preprocessor.impute_values['annual_income'] == 60000.0
    assert preprocessor.impute_values['age'] == 40.0
    
    # Transform
    df_transformed = preprocessor.transform(df_train)
    
    # Check that there are no NaNs in transformed output
    assert df_transformed.isna().sum().sum() == 0
    
    # Check that payment_history was mapped to float numeric values
    assert pd.api.types.is_float_dtype(df_transformed['payment_history'])
    
    # Check that numerical columns are standard scaled (mean ~ 0)
    # Mean of standard scaled values should be very close to 0
    assert np.allclose(df_transformed['annual_income'].mean(), 0, atol=1e-5)

def test_feature_engineering():
    df = pd.DataFrame({
        'annual_income': [60000.0],
        'monthly_income': [5000.0],
        'loan_amount': [20000.0],
        'existing_debts': [10000.0],
        'debt_to_income_ratio': [0.1667],
        'number_of_credit_cards': [2],
        'credit_utilization_ratio': [0.4],
        'payment_history': ['Good'],
        'number_of_late_payments': [0],
        'loan_repayment_history': ['Mostly Paid'],
        'employment_length': [4.0],
        'age': [34],
        'savings_balance': [8000.0],
        'previous_defaults': ['No'],
        'credit_history_length': [8.0]
    })
    
    df_engineered = add_derived_features(df)
    
    # Verify derived features are present
    assert 'debt_to_income_ratio_refined' in df_engineered.columns
    assert 'credit_utilization_percentage' in df_engineered.columns
    assert 'payment_consistency_score' in df_engineered.columns
    assert 'financial_stability_score' in df_engineered.columns
    assert 'risk_index' in df_engineered.columns
    assert 'credit_age_score' in df_engineered.columns
    
    # Verify values
    assert df_engineered['credit_utilization_percentage'].iloc[0] == 40.0
    # credit_age_score = credit_history / (age - 17) = 8 / 17 = 0.4706
    assert abs(df_engineered['credit_age_score'].iloc[0] - 0.4706) < 1e-3
