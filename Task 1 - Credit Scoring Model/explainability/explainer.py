import os
import pickle
import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class CreditExplainer:
    def __init__(self, model, background_data: pd.DataFrame):
        self.model = model
        self.feature_names = background_data.columns.tolist()
        
        # Build the best-suited SHAP explainer
        model_name = type(model).__name__
        logger.info(f"Building SHAP explainer for model type: {model_name}")
        
        try:
            if 'XGB' in model_name or 'Forest' in model_name or 'Boosting' in model_name or 'DecisionTree' in model_name:
                self.explainer = shap.TreeExplainer(model)
            elif 'Logistic' in model_name:
                self.explainer = shap.LinearExplainer(model, background_data)
            else:
                self.explainer = shap.Explainer(model, background_data)
        except Exception as e:
            logger.warning(f"Failed to initialize specialized explainer, falling back to default Explainer. Error: {e}")
            self.explainer = shap.Explainer(model, background_data)

    def explain_instance(self, instance_df: pd.DataFrame) -> dict:
        """
        Compute SHAP values for a single instance.
        Returns a dictionary of {feature_name: shap_value} representing contribution to the positive class (Good Credit).
        """
        # Ensure correct column ordering
        instance_df = instance_df[self.feature_names]
        
        # Compute shap values
        shap_res = self.explainer(instance_df)
        
        # Handle different output formats of different explainers
        # For Binary Classification, we care about the probability of Class 1 (Good Credit)
        raw_vals = shap_res.values[0]
        
        # Check if shap output has 3 dimensions or list for classes
        if isinstance(raw_vals, np.ndarray) and len(raw_vals.shape) == 2 and raw_vals.shape[1] == 2:
            # Multi-class format (class 0, class 1). We take class 1.
            raw_vals = raw_vals[:, 1]
        elif isinstance(raw_vals, list) or (isinstance(raw_vals, np.ndarray) and len(raw_vals.shape) > 1 and raw_vals.shape[-1] == 2):
            raw_vals = raw_vals[..., 1]
            
        # Create dict mapping feature name to its contribution
        explanation = {}
        for name, val in zip(self.feature_names, raw_vals):
            explanation[name] = float(val)
            
        return explanation

    def plot_local_explanation(self, instance_df: pd.DataFrame, save_path: str):
        """
        Plot local SHAP explanation (waterfall/bar plot) and save to disk.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Reorder
        instance_df = instance_df[self.feature_names]
        
        shap_values = self.explainer(instance_df)
        
        # Extract base value and values for class 1 if multi-class
        base_value = shap_values.base_values[0]
        values = shap_values.values[0]
        data = shap_values.data[0]
        
        if isinstance(base_value, (np.ndarray, list)) and len(base_value) == 2:
            base_value = base_value[1]
            values = values[:, 1] if len(values.shape) == 2 else values[1]
            
        # Re-build simple Explainer object for plotting
        exp_obj = shap.Explanation(
            values=values,
            base_values=base_value,
            data=data,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        # Draw a bar plot of contributions
        shap.plots.bar(exp_obj, show=False)
        plt.title("Individual Prediction Explanation (SHAP)", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"SHAP local explanation plot saved to {save_path}")

    def save(self, filepath: str):
        """Save the explainer to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"SHAP explainer saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'CreditExplainer':
        """Load explainer from file."""
        with open(filepath, 'rb') as f:
            explainer = pickle.load(f)
        logger.info(f"SHAP explainer loaded from {filepath}")
        return explainer
