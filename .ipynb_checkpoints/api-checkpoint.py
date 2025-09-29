#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# versión para docker -*
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path

PORT = int(os.environ.get("PORT", "8080"))
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_breastcancer.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.pkl")

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Nombres de las características esperadas (del dataset original)
EXPECTED_FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points',
    'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error',
    'perimeter error', 'area error', 'smoothness error', 'compactness error',
    'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points',
    'worst symmetry', 'worst fractal dimension'
]

@app.route("/", methods=['GET'])
def home():
    return jsonify({
        "message": "API de Clasificación de Breast Cancer",
        "status": "active",
        "endpoints": {
            "health_check": "/health (GET)",
            "prediction": "/predict (POST)"
        },
        "expected_features": EXPECTED_FEATURES,
        "num_features_expected": len(EXPECTED_FEATURES)
    })

@app.route("/health", methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "message": "Servicio activo y modelo cargado",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get("features")
    
    if not features or not isinstance(features, list):
        return jsonify({
            "error": "La clave 'features' debe ser una lista de listas.",
            "expected_format": {
                "features": "Lista de muestras, cada muestra debe tener 30 características numéricas"
            }
        }), 400
    
    try:
        # Validar que cada muestra tenga 30 características
        for i, sample in enumerate(features):
            if len(sample) != len(EXPECTED_FEATURES):
                return jsonify({
                    "error": f"Cada muestra debe contener {len(EXPECTED_FEATURES)} features. Muestra {i} tiene {len(sample)}",
                    "expected_features_count": len(EXPECTED_FEATURES),
                    "received_features_count": len(sample)
                }), 400
        
        # Convertir a numpy array
        X = np.array(features, dtype=float)
        
        # ESCALAR LOS DATOS (igual que en entrenamiento)
        X_scaled = scaler.transform(X)
        
        # Realizar predicciones
        predictions = model.predict(X_scaled).tolist()
        
        # Mapear predicciones a nombres de clases
        class_names = ['malignant', 'benign']  # 0: malignant, 1: benign
        named_predictions = [class_names[p] for p in predictions]
        
        # Probabilidades (opcional)
        probabilities = model.predict_proba(X_scaled).tolist()
        
        return jsonify({
            "predictions": named_predictions,
            "probabilities": probabilities,
            "class_mapping": {
                "0": "malignant",
                "1": "benign"
            }
        })

    except (ValueError, TypeError) as e:
        return jsonify({
            "error": f"Error en el formato de los datos de entrada: {str(e)}",
            "expected_features": EXPECTED_FEATURES
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)

