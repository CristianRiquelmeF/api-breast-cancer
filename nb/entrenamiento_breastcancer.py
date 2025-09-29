#!/usr/bin/env python
# coding: utf-8

# ## Carga de datos

# In[1]:


# Librerías principales

import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# In[2]:


# Cargar el dataset
cancer_data = load_breast_cancer()
print(cancer_data.keys())


# In[3]:


# Crear un DataFrame
df_features = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df_target = pd.DataFrame(cancer_data.target, columns=['target'])

# Unir ambos DataFrames en uno solo
df = pd.concat([df_features, df_target], axis=1)


# In[4]:


df.head(3)


# In[5]:


df.info()


# In[8]:


# Conteo de clases en variable objetivo
df['target'].value_counts()


# In[7]:


# Mostrar estadísticas descriptivas
print("\nResumen estadístico de las características:")
df.describe()


# ## Preprocesamiento

# In[9]:


# Separar las características (X) del objetivo (y)
X = df.drop('target', axis=1)
y = df['target']

print("Dimensiones de X (características):", X.shape)
print("Dimensiones de y (objetivo):", y.shape)


# In[10]:


# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

print("Tamaño del conjunto de entrenamiento:", X_train.shape[0], "muestras")
print("Tamaño del conjunto de prueba:", X_test.shape[0], "muestras")


# In[11]:


# Inicializar el escalador
scaler = StandardScaler()

# Ajustar el escalador con los datos de entrenamiento y transformarlos
X_train_scaled = scaler.fit_transform(X_train)

# Transformar los datos de prueba usando el mismo escalador
X_test_scaled = scaler.transform(X_test)

# Convertir los arrays de numpy a DataFrames para visualizarlos (opcional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


# ## Modelamiento

# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)


# - **Random Forest con Grid Search**

# In[13]:


# Definir la parrilla de hiperparámetros a probar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]}

# Inicializar el modelo RandomForest
rf = RandomForestClassifier(random_state=42)

# Inicializar GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')


# In[14]:


# Entrenar GridSearchCV para encontrar los mejores parámetros
grid_search.fit(X_train_scaled, y_train)


# In[15]:


# Obtener los mejores parámetros encontrados y guardar
best_params = grid_search.best_params_
print(f"\nMejores hiperparámetros encontrados:\n{best_params}")

best_rf_model = grid_search.best_estimator_

# Realizar predicciones en el conjunto de prueba
y_pred = best_rf_model.predict(X_test_scaled)


# In[16]:


# Métricas de evaluación (multiclase)
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
    "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),}

print("Métricas en test:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=cancer_data.target_names))


# ## Guardar Artefactos del Modelo

# In[17]:


# Crearun directorio 'artifacts' para guardar archivos
from pathlib import Path
ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)


# - **Serialización del modelo**

# In[18]:


# Guardar el modelo en ruta creada
model_filename = 'modelo_breastcancer.pkl'
full_model_path = ARTIFACTS_DIR / model_filename
joblib.dump(best_rf_model, full_model_path)


# - **Scaler utilizado**
# 
# Las predicciones deben usar datos escalados de la misma manera que fue entrenado. Cuando se envían nuevos datos a la API para hacer una predicción, estos datos vienen en su escala original, si se pasan directamente al modelo, él los interpretará incorrectamente porque no están en la escala que espera.

# In[19]:


# Guardar el scaler
scaler_filename = 'scaler.pkl'
full_scaler_path = ARTIFACTS_DIR / scaler_filename
joblib.dump(scaler, full_scaler_path)


# - Manifiesto/Model Card

# In[20]:


import time
import platform
import json


# In[21]:


# Definir manifiesto para guardar en JSON
manifest = {
    "name": "RandomForest-BreastCancer",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "framework": "scikit-learn",
    "sklearn_version": sklearn.__version__,
    "python_version": platform.python_version(),
    "features": list(X.columns),
    "target": y.name,
    "scaler_info": {
    "type": "StandardScaler",
    "fitted_on_training_data": True,
    "scaler_path": str(full_scaler_path)},
    "best_params": grid_search.best_params_,
    "cv_metric": grid_search.scoring,
    "cv_best_score": float(grid_search.best_score_),
    "test_metrics": metrics,}

manifest_path = ARTIFACTS_DIR / "model_manifest.json"

# Escribimos el diccionario en el archivo JSON con un formato legible.
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Manifest del modelo guardado en: {manifest_path.resolve()}")

