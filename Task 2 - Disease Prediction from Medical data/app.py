import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')

# Cache for loaded models and scalers to optimize prediction performance
MODELS_CACHE = {}
SCALERS_CACHE = {}

def get_model_and_scaler(disease, algorithm):
    model_key = f"{disease}_{algorithm}"
    scaler_key = disease
    
    if model_key not in MODELS_CACHE:
        model_path = f"models/{disease}_{algorithm}.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        MODELS_CACHE[model_key] = joblib.load(model_path)
        
    if scaler_key not in SCALERS_CACHE:
        scaler_path = f"models/{disease}_scaler.joblib"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        SCALERS_CACHE[scaler_key] = joblib.load(scaler_path)
        
    return MODELS_CACHE[model_key], SCALERS_CACHE[scaler_key]

# Load metrics configuration to verify feature orders
METRICS_DATA = None
def get_metrics_data():
    global METRICS_DATA
    if METRICS_DATA is None:
        metrics_path = 'static/model_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                METRICS_DATA = json.load(f)
        else:
            METRICS_DATA = {}
    return METRICS_DATA

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics = get_metrics_data()
    if not metrics:
        return jsonify({"success": False, "error": "Metrics file not found"}), 404
    return jsonify({"success": True, "data": metrics})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No input data provided"}), 400
            
        disease = data.get('disease')
        algorithm = data.get('algorithm')
        input_features = data.get('features')
        
        if not all([disease, algorithm, input_features]):
            return jsonify({"success": False, "error": "Missing disease, algorithm, or features parameter"}), 400
            
        # Verify model details against metrics
        metrics = get_metrics_data()
        if disease not in metrics:
            return jsonify({"success": False, "error": f"Unsupported disease: {disease}"}), 400
            
        disease_meta = metrics[disease]
        if algorithm not in disease_meta['models']:
            return jsonify({"success": False, "error": f"Unsupported algorithm: {algorithm}"}), 400
            
        required_features = disease_meta['features']
        
        # Parse and order features correctly
        ordered_values = []
        for feat in required_features:
            if feat not in input_features:
                return jsonify({"success": False, "error": f"Missing required feature: {feat}"}), 400
            try:
                ordered_values.append(float(input_features[feat]))
            except ValueError:
                return jsonify({"success": False, "error": f"Invalid numeric value for feature: {feat}"}), 400
                
        # Load model and scaler
        try:
            model, scaler = get_model_and_scaler(disease, algorithm)
        except FileNotFoundError as e:
            return jsonify({"success": False, "error": str(e)}), 500
            
        # Preprocess input (Scale features)
        input_df = pd.DataFrame([ordered_values], columns=required_features)
        scaled_input = scaler.transform(input_df)
        
        # Predict class and probabilities
        prediction = int(model.predict(scaled_input)[0])
        probabilities = model.predict_proba(scaled_input)[0]
        
        # Probability of class 1 (at risk)
        # Class 1 is usually the second class in binary classification
        risk_probability = float(probabilities[1])
        
        # Define risk levels
        if risk_probability < 0.35:
            risk_level = "Low"
        elif risk_probability < 0.70:
            risk_level = "Moderate"
        else:
            risk_level = "High"
            
        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": risk_probability,
            "risk_level": risk_level,
            "message": "Prediction computed successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Use standard host and port
    app.run(host='127.0.0.1', port=5000, debug=True)
