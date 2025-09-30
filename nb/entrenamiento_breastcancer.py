#!/usr/bin/env python
# coding: utf-8

# 
# **Proyecto: Servicio de Diagnóstico Predictivo**
# - Nombre: Cristian Riquelme F.
# 
# **Contexto y objetivo:**
# 
# Este proyecto aborda el desarrollo de un sistema de Machine Learning de extremo a extremo para la clasificación de tumores de mama, enmarcado en las necesidades de una startup de tecnología para la salud (Health-Tech). El objetivo principal es construir un servicio de diagnóstico preventivo que sea robusto, escalable, y fácil de mantener, siguiendo las mejores prácticas de MLOps.
# 
# El sistema completo integrará un modelo de clasificación, lo expondrá a través de una API REST, será distribuido como un contenedor Docker y contará con un flujo de Integración y Despliegue Continuo (CI/CD) para automatizar las pruebas y actualizaciones.

# ## Carga de datos

# Conjunto de datos "Wisconsin Breast Cancer", una base de datos estándar y bien conocida para problemas de clasificación binaria en el diagnóstico de cáncer.

# In[12]:


# Librerías principales

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# In[5]:


# Cargar el dataset
cancer_data = load_breast_cancer()
print(cancer_data.keys())


# In[6]:


# Crear un DataFrame
df_features = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df_target = pd.DataFrame(cancer_data.target, columns=['target'])

# Unir ambos DataFrames en uno solo
df = pd.concat([df_features, df_target], axis=1)


# In[7]:


df.head(3)


# In[22]:


df.info()


# In[13]:


# Conteo de clases en variable objetivo
df['target'].value_counts()


# In[14]:


# Mostrar estadísticas descriptivas
print("\nResumen estadístico de las características:")
df.describe()


# - **Gráfico de barras de la variable objetivo**

# In[19]:


# Crear una figura para el gráfico
plt.figure(figsize=(6, 4))

# Crear el gráfico de conteo
ax = sns.countplot(x='target', data=df, palette='viridis')

# Añadir título y etiquetas
plt.title('Distribución de Diagnósticos (Target)', fontsize=16)
plt.xlabel('Diagnóstico', fontsize=12)
plt.ylabel('Cantidad de Casos', fontsize=12)

# Cambiar las etiquetas del eje x para mayor claridad
ax.set_xticklabels(['Maligno (0)', 'Benigno (1)'])

# Mostrar el gráfico
plt.show()


# - **Heatmap**
# 
# Un heatmap o mapa de calor nos permite identificar rápidamente qué características están fuertemente correlacionadas entre sí. Esto es útil para detectar multicolinealidad.
# Solo se usarán las 10 primeras características de "mean" para que el gráfico sea más legible.
# 
# Mean radius, mean perimeter y mean area están casi perfectamente correlacionadas, lo cual es lógico, ya que son medidas geométricas interdependientes.

# In[21]:


# Seleccionar las primeras 10 características (las que tienen "mean")
mean_features = df.columns[0:10]
corr_matrix = df[mean_features].corr()

# Crear una figura de mayor tamaño para que se vea bien
plt.figure(figsize=(10, 8))

# Crear el heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Añadir título
plt.title('Mapa de Calor de Correlación (Características "Mean")', fontsize=16)

# Mostrar el gráfico
plt.show()


# In[ ]:





# ## Preprocesamiento

# In[23]:


# Separar las características (X) del objetivo (y)
X = df.drop('target', axis=1)
y = df['target']

print("Dimensiones de X (características):", X.shape)
print("Dimensiones de y (objetivo):", y.shape)


# In[24]:


# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

print("Tamaño del conjunto de entrenamiento:", X_train.shape[0], "muestras")
print("Tamaño del conjunto de prueba:", X_test.shape[0], "muestras")


# In[25]:


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

# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,RocCurveDisplay)


# - **Random Forest con Grid Search**
# 
# Se entrena un modelo Random Forest Classifier. Para asegurar un rendimiento óptimo, se implementa una búsqueda exhaustiva de hiperparámetros (GridSearchCV), evaluando múltiples combinaciones para encontrar la que ofrece la mayor precisión.

# In[27]:


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


# In[28]:


# Entrenar GridSearchCV para encontrar los mejores parámetros
grid_search.fit(X_train_scaled, y_train)


# In[30]:


# Obtener los mejores parámetros encontrados y guardar
best_params = grid_search.best_params_
print(f"\nMejores hiperparámetros encontrados:\n{best_params}")

best_rf_model = grid_search.best_estimator_

# Realizar predicciones en el conjunto de prueba
y_pred = best_rf_model.predict(X_test_scaled)


# In[31]:


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


# In[33]:


# Gráfica curva ROC
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    best_rf_model, 
    X_test_scaled, 
    y_test,
    name=f"Random Forest (AUC = {best_rf_model.score(X_test_scaled, y_test):.2f})",
    color="darkorange")
plt.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio")
plt.legend(loc="lower right")
plt.title('Curva ROC - Random Forest')
plt.grid(True)
plt.show()


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


# In[22]:


print("\nContenido del Manifiesto:")
print(json.dumps(manifest, indent=2))


# In[25]:


# Convertir a script
get_ipython().system('jupyter nbconvert --to script entrenamiento_breastcancer.ipynb')

