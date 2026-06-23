import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, numerical_cols, categorical_cols, target_col=None):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        
        # State variables to fit
        self.impute_values = {}
        self.outlier_bounds = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Explicit ordinal mapping for categorical columns
        self.categorical_mappings = {
            'payment_history': {
                'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3,
                0: 0, 1: 1, 2: 2, 3: 3  # support pre-mapped
            },
            'loan_repayment_history': {
                'Defaulted': 0, 'Delayed': 1, 'Mostly Paid': 2, 'All Paid': 3,
                0: 0, 1: 1, 2: 2, 3: 3  # support pre-mapped
            },
            'previous_defaults': {
                'No': 0, 'Yes': 1,
                0: 0, 1: 1  # support pre-mapped
            }
        }
        
    def fit(self, df: pd.DataFrame):
        """Fit preprocessor parameters on training data."""
        logger.info("Fitting Preprocessor...")
        df_copy = df.copy()
        
        # 1. Learn Imputation values
        for col in self.numerical_cols:
            self.impute_values[col] = float(df_copy[col].median())
            
        for col in self.categorical_cols:
            # Mode returns a Series, take first element
            mode_val = df_copy[col].mode()
            self.impute_values[col] = mode_val.iloc[0] if not mode_val.empty else "Good"
            
        # 2. Learn Outlier limits (1st and 99th percentiles)
        for col in self.numerical_cols:
            q_low = float(df_copy[col].quantile(0.01))
            q_high = float(df_copy[col].quantile(0.99))
            self.outlier_bounds[col] = (q_low, q_high)
            
        # Temporarily apply imputation & outlier capping for scaler fitting
        imputed_df = self._impute_and_cap(df_copy)
        
        # Encode categoricals before scaling them or keeping them separate
        # Note: we fit the scaler ONLY on numerical columns + encoded categoricals.
        # Actually, let's encode the categoricals and fit the scaler only on numeric columns.
        # Let's scale numerical columns only. Encoded categoricals can stay in their 0-3 range.
        numeric_df = imputed_df[self.numerical_cols]
        self.scaler.fit(numeric_df)
        
        self.is_fitted = True
        logger.info("Preprocessor fitting completed successfully.")
        return self
        
    def _impute_and_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal helper to fill NAs and winsorize outliers."""
        df_out = df.copy()
        
        # Impute numerical columns
        for col in self.numerical_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(self.impute_values[col])
                
        # Impute categorical columns
        for col in self.categorical_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(self.impute_values[col])
                
        # Cap outliers for numerical features
        for col in self.numerical_cols:
            if col in df_out.columns:
                q_low, q_high = self.outlier_bounds[col]
                df_out[col] = df_out[col].clip(lower=q_low, upper=q_high)
                
        return df_out
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to a DataFrame."""
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet!")
            
        df_out = df.copy()
        
        # 1. Impute and Cap outliers
        df_out = self._impute_and_cap(df_out)
        
        # 2. Encode categoricals explicitly
        for col in self.categorical_cols:
            if col in df_out.columns:
                mapping = self.categorical_mappings.get(col, {})
                # Fill values not in mapping with the most common mapping (imputed)
                default_mapped_val = mapping.get(self.impute_values[col], 0)
                df_out[col] = df_out[col].map(lambda x: mapping.get(x, default_mapped_val)).astype(float)
                
        # 3. Scale numerical features
        if self.numerical_cols:
            scaled_vals = self.scaler.transform(df_out[self.numerical_cols])
            df_out[self.numerical_cols] = scaled_vals
            
        # 4. Handle target column if present
        if self.target_col and self.target_col in df_out.columns:
            target_mapping = {'Good': 1.0, 'Bad': 0.0, 1: 1.0, 0: 0.0, 1.0: 1.0, 0.0: 0.0}
            df_out[self.target_col] = df_out[self.target_col].map(target_mapping).astype(float)
            
        return df_out

    def save(self, filepath: str):
        """Save the preprocessor to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {filepath}")
        
    @staticmethod
    def load(filepath: str) -> 'Preprocessor':
        """Load preprocessor from file."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
