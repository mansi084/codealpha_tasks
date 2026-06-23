import os
import yaml
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=2500, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Base features
    age = np.random.randint(18, 76, size=num_samples)
    
    # Employment length is bounded by age (cannot start working before 16, typically)
    employment_length = np.array([
        max(0, np.random.randint(0, max(1, a - 16)))
        for a in age
    ])
    # Add random float noise
    employment_length = np.round(employment_length + np.random.uniform(0, 1, size=num_samples), 1)
    
    annual_income = np.random.exponential(scale=65000, size=num_samples) + 20000
    annual_income = np.round(np.clip(annual_income, 15000, 300000), -2)
    
    monthly_income = np.round((annual_income / 12.0) + np.random.normal(0, 200, size=num_samples), 2)
    monthly_income = np.clip(monthly_income, 1000, None)
    
    # Savings balance: higher income generally correlates with higher savings
    savings_balance = np.random.exponential(scale=0.2 * annual_income) + 500
    savings_balance = np.round(np.clip(savings_balance, 0, 200000), -2)
    
    # Loan amount
    loan_amount = np.random.exponential(scale=35000, size=num_samples) + 5000
    loan_amount = np.round(np.clip(loan_amount, 3000, 150000), -2)
    
    # Existing debts: correlated with income and loan amount
    existing_debts = np.random.exponential(scale=0.3 * annual_income)
    existing_debts = np.round(np.clip(existing_debts, 0, 120000), -2)
    
    debt_to_income_ratio = np.round(existing_debts / annual_income, 4)
    
    number_of_credit_cards = np.random.poisson(lam=3, size=num_samples)
    number_of_credit_cards = np.clip(number_of_credit_cards, 0, 12)
    
    # Credit utilization ratio
    credit_utilization_ratio = np.random.beta(a=2, b=5, size=num_samples)
    # High utilization for people with many cards/higher debt
    util_modifier = np.clip(existing_debts / (annual_income + 1e-5), 0, 0.5)
    credit_utilization_ratio = np.round(np.clip(credit_utilization_ratio + util_modifier, 0.0, 1.0), 3)
    
    # Credit history length: correlated with age and employment length
    credit_history_length = np.array([
        max(0.5, np.round(np.random.uniform(0.1, 0.8) * (a - 17), 1))
        for a in age
    ])
    
    # Late payments
    number_of_late_payments = np.random.poisson(lam=0.8, size=num_samples)
    # Scale late payments with credit utilization and poor habits
    number_of_late_payments = np.clip(number_of_late_payments + (credit_utilization_ratio * 4).astype(int), 0, 15)
    
    # Categoricals
    payment_hist_probs = np.array([
        [0.7, 0.2, 0.08, 0.02] if l == 0 else
        [0.2, 0.5, 0.2, 0.1] if l <= 2 else
        [0.05, 0.15, 0.5, 0.3]
        for l in number_of_late_payments
    ])
    
    payment_history_cats = ["Excellent", "Good", "Fair", "Poor"]
    payment_history = [
        np.random.choice(payment_history_cats, p=probs)
        for probs in payment_hist_probs
    ]
    
    previous_defaults_probs = [
        0.02 if l == 0 else 0.1 if l <= 2 else 0.4
        for l in number_of_late_payments
    ]
    previous_defaults = [
        "Yes" if np.random.rand() < p else "No"
        for p in previous_defaults_probs
    ]
    
    # Loan repayment history
    repay_hist_cats = ["All Paid", "Mostly Paid", "Delayed", "Defaulted"]
    loan_repay_probs = []
    for i in range(num_samples):
        has_default = previous_defaults[i] == "Yes"
        late_cnt = number_of_late_payments[i]
        
        if has_default:
            probs = [0.05, 0.15, 0.3, 0.5]
        elif late_cnt >= 4:
            probs = [0.1, 0.3, 0.4, 0.2]
        elif late_cnt >= 1:
            probs = [0.2, 0.5, 0.2, 0.1]
        else:
            probs = [0.7, 0.25, 0.05, 0.0]
        loan_repay_probs.append(probs)
        
    loan_repayment_history = [
        np.random.choice(repay_hist_cats, p=probs)
        for probs in loan_repay_probs
    ]
    
    # 2. Build target logic (probability of bad credit risk)
    # Define scoring function
    score = np.zeros(num_samples)
    
    # Base risk factor
    score -= 1.8
    
    # High late payments increases risk
    score += (number_of_late_payments * 0.45)
    
    # Previous defaults is a huge risk factor
    score += np.where(np.array(previous_defaults) == "Yes", 2.2, 0.0)
    
    # High credit utilization increases risk
    score += (credit_utilization_ratio * 2.5)
    
    # Poor payment history increase risk
    pay_history_arr = np.array(payment_history)
    score += np.where(pay_history_arr == "Poor", 1.8, 0.0)
    score += np.where(pay_history_arr == "Fair", 0.9, 0.0)
    score -= np.where(pay_history_arr == "Excellent", 0.8, 0.0)
    
    # Poor loan repayment history increases risk
    repay_arr = np.array(loan_repayment_history)
    score += np.where(repay_arr == "Defaulted", 2.5, 0.0)
    score += np.where(repay_arr == "Delayed", 1.2, 0.0)
    score -= np.where(repay_arr == "All Paid", 0.6, 0.0)
    
    # Low income/savings increases risk
    score -= np.log(annual_income / 15000) * 0.7
    score -= np.log(savings_balance + 200) * 0.4
    
    # Higher debt-to-income ratio increases risk
    score += (debt_to_income_ratio * 2.8)
    
    # Higher loan amount vs income increases risk
    score += (loan_amount / annual_income) * 1.5
    
    # Short credit history/employment increases risk
    score -= (credit_history_length / 12.0) * 0.4
    score -= (employment_length / 8.0) * 0.3
    
    # Calculate probability of being "Bad" using logistic function
    prob_bad = 1.0 / (1.0 + np.exp(-score))
    
    # Assign target variable
    random_draw = np.random.rand(num_samples)
    creditworthiness = np.where(random_draw < prob_bad, "Bad", "Good")
    
    # Build dataframe
    df = pd.DataFrame({
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "existing_debts": existing_debts,
        "debt_to_income_ratio": debt_to_income_ratio,
        "number_of_credit_cards": number_of_credit_cards,
        "credit_utilization_ratio": credit_utilization_ratio,
        "payment_history": payment_history,
        "number_of_late_payments": number_of_late_payments,
        "loan_repayment_history": loan_repayment_history,
        "employment_length": employment_length,
        "age": age,
        "savings_balance": savings_balance,
        "previous_defaults": previous_defaults,
        "credit_history_length": credit_history_length,
        "creditworthiness": creditworthiness
    })
    
    # Add a tiny amount of random NaNs to test preprocessing missing value handling
    for col in ["savings_balance", "employment_length", "credit_utilization_ratio"]:
        mask = np.random.rand(num_samples) < 0.02
        df.loc[mask, col] = np.nan
        
    # Categorical NaNs
    mask = np.random.rand(num_samples) < 0.01
    df.loc[mask, "payment_history"] = np.nan
        
    return df

if __name__ == "__main__":
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    
    # Load config yaml for raw data path
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    output_path = config['paths']['raw_data']
    print(f"Generating synthetic credit scoring dataset...")
    df = generate_synthetic_data(num_samples=2500)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully to {output_path} with shape {df.shape}")
    print(f"Creditworthiness Distribution:\n{df['creditworthiness'].value_counts(dropna=False)}")
