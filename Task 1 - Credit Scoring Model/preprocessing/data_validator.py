import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, expected_numerical_features, expected_categorical_features, target_col=None):
        self.numerical_cols = expected_numerical_features
        self.categorical_cols = expected_categorical_features
        self.target_col = target_col
        
    def validate(self, df: pd.DataFrame, is_training: bool = False) -> tuple[bool, list[str]]:
        """
        Validate pandas DataFrame schema, types, and values.
        Returns:
            (is_valid: bool, errors: list of strings)
        """
        errors = []
        
        # 1. Check columns existence
        expected_cols = self.numerical_cols + self.categorical_cols
        if is_training and self.target_col:
            expected_cols = expected_cols + [self.target_col]
            
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
            logger.error(f"Validation failed: missing columns {missing_cols}")
            return False, errors
            
        # 2. Check types & constraints
        for col in self.numerical_cols:
            # Drop null values for range checking, handled in preprocessor later
            series = df[col].dropna()
            
            # Numeric conversion check
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric.")
                
            # Business logic constraints
            if col in ['annual_income', 'monthly_income', 'loan_amount', 'existing_debts', 'savings_balance']:
                if (series < 0).any():
                    errors.append(f"Column '{col}' contains negative values.")
            
            if col in ['age']:
                if (series < 18).any() or (series > 120).any():
                    errors.append(f"Column 'age' contains values out of realistic range (18-120).")
                    
            if col in ['credit_utilization_ratio', 'debt_to_income_ratio']:
                # Utilization ratio can sometimes exceed 1.0 (over limit), but shouldn't be negative
                if (series < 0).any():
                    errors.append(f"Column '{col}' contains negative values.")
                    
            if col in ['number_of_credit_cards', 'number_of_late_payments', 'employment_length', 'credit_history_length']:
                if (series < 0).any():
                    errors.append(f"Column '{col}' contains negative values.")
                    
        # 3. Check categorical values
        valid_payment_history = {"Excellent", "Good", "Fair", "Poor"}
        valid_loan_repay = {"All Paid", "Mostly Paid", "Delayed", "Defaulted"}
        valid_defaults = {"Yes", "No"}
        
        if 'payment_history' in df.columns:
            invalid_vals = df['payment_history'].dropna().unique()
            invalid_vals = [v for v in invalid_vals if v not in valid_payment_history]
            if invalid_vals:
                errors.append(f"Column 'payment_history' has invalid values: {invalid_vals}")
                
        if 'loan_repayment_history' in df.columns:
            invalid_vals = df['loan_repayment_history'].dropna().unique()
            invalid_vals = [v for v in invalid_vals if v not in valid_loan_repay]
            if invalid_vals:
                errors.append(f"Column 'loan_repayment_history' has invalid values: {invalid_vals}")
                
        if 'previous_defaults' in df.columns:
            invalid_vals = df['previous_defaults'].dropna().unique()
            invalid_vals = [v for v in invalid_vals if v not in valid_defaults]
            if invalid_vals:
                errors.append(f"Column 'previous_defaults' has invalid values: {invalid_vals}")
                
        if is_training and self.target_col and self.target_col in df.columns:
            valid_targets = {"Good", "Bad"}
            invalid_vals = [v for v in df[self.target_col].unique() if v not in valid_targets]
            if invalid_vals:
                errors.append(f"Target column '{self.target_col}' has invalid values: {invalid_vals}")
                
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Data validation succeeded.")
        else:
            logger.warning(f"Data validation failed with {len(errors)} errors.")
            
        return is_valid, errors
