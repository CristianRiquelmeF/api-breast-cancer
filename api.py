# versión para docker -*
import os
import joblib
import numpy as np
import pandas as pd
import json
from flask import Flask, request, jsonify
from pathlib import Path

PORT = int(os.environ.get("PORT", "8080"))
MODEL_PATH = "artifacts/modelo_breastcancer.pkl"
SCALER_PATH = "artifacts/scaler.pkl"
MANIFEST_PATH = "artifacts/model_manifest.json"

app = Flask(__name__)

# Cargar modelo, scaler y manifiesto
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Cargar manifiesto del modelo
try:
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    manifest_loaded = True
except FileNotFoundError:
    manifest = {"error": "Manifiesto no encontrado"}
    manifest_loaded = False

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
            "model_info": "/model-info (GET)",
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
        "scaler_loaded": scaler is not None,
        "manifest_loaded": manifest_loaded
    })

@app.route("/model-info", methods=['GET'])
def model_info():
    """Endpoint para mostrar los metadatos del modelo desde el manifiesto"""
    if not manifest_loaded:
        return jsonify({
            "error": "No se pudo cargar el manifiesto del modelo",
            "message": "El archivo model_manifest.json no se encuentra en la ruta especificada"
        }), 404
    
    # Filtrar información relevante para mostrar
    model_info_response = {
        "name": manifest.get("name", "N/A"),
        "created_at": manifest.get("created_at", "N/A"),
        "framework": manifest.get("framework", "N/A"),
        "model_type": "RandomForestClassifier",
        "target_variable": manifest.get("target", "N/A"),
        "target_classes": {
            "0": "malignant",
            "1": "benign"
        },
        "features_count": len(manifest.get("features", [])),
        "features": manifest.get("features", []),
        "scaler_info": manifest.get("scaler_info", {}),
        "best_hyperparameters": manifest.get("best_params", {}),
        "cross_validation": {
            "metric": manifest.get("cv_metric", "N/A"),
            "best_score": manifest.get("cv_best_score", "N/A")
        },
        "test_performance": manifest.get("test_metrics", {}),
        "training_info": {
            "python_version": manifest.get("python_version", "N/A"),
            "sklearn_version": manifest.get("sklearn_version", "N/A")
        }
    }
    
    return jsonify(model_info_response)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get("features")
    
    if not features:
        return jsonify({
            "error": "La clave 'features' es requerida.",
            "expected_format": {
                "features": "Lista de 30 valores numéricos (predicción individual) o lista de listas (predicción por lote)"
            }
        }), 400
    
    try:
        # Verificar si es predicción individual (lista plana) o por lote (lista de listas)
        if isinstance(features, list) and len(features) > 0:
            if isinstance(features[0], list):
                # Predicción por lote
                for i, sample in enumerate(features):
                    if len(sample) != len(EXPECTED_FEATURES):
                        return jsonify({
                            "error": f"Cada muestra debe contener {len(EXPECTED_FEATURES)} features. Muestra {i} tiene {len(sample)}",
                            "expected_features_count": len(EXPECTED_FEATURES),
                            "received_features_count": len(sample)
                        }), 400
                X = np.array(features, dtype=float)
            else:
                # Predicción individual - lista plana de 30 características
                if len(features) != len(EXPECTED_FEATURES):
                    return jsonify({
                        "error": f"Se esperaban {len(EXPECTED_FEATURES)} características. Se recibieron {len(features)}",
                        "expected_features_count": len(EXPECTED_FEATURES),
                        "received_features_count": len(features)
                    }), 400
                X = np.array([features], dtype=float)  # Convertir a 2D array
        else:
            return jsonify({
                "error": "El formato de 'features' es inválido. Debe ser una lista de 30 valores o una lista de listas."
            }), 400
        
        # Crear DataFrame con nombres de características para evitar warnings
        X_df = pd.DataFrame(X, columns=EXPECTED_FEATURES)
        
        # ESCALAR LOS DATOS (igual que en entrenamiento)
        X_scaled = scaler.transform(X_df)
        
        # Realizar predicciones
        predictions = model.predict(X_scaled).tolist()
        
        # Mapear predicciones a nombres de clases
        class_names = ['malignant', 'benign']  # 0: malignant, 1: benign
        named_predictions = [class_names[p] for p in predictions]
        
        # Probabilidades (opcional)
        probabilities = model.predict_proba(X_scaled).tolist()
        
        # Para predicción individual, devolver resultado simple
        if X.shape[0] == 1:
            return jsonify({
                "prediction": named_predictions[0],
                "class": predictions[0],
                "probabilities": {
                    "malignant": probabilities[0][0],
                    "benign": probabilities[0][1]
                },
                "class_mapping": {
                    "0": "malignant",
                    "1": "benign"
                }
            })
        else:
            # Para predicción por lote
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "sample": i,
                    "prediction": named_predictions[i],
                    "class": pred,
                    "probabilities": {
                        "malignant": prob[0],
                        "benign": prob[1]
                    }
                })
            
            return jsonify({
                "predictions": results,
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