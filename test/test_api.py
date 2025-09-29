#!/usr/bin/env python
# coding: utf-8

# **PRUEBA DE API REST CANCER BREAST**

# In[6]:


#!/usr/bin/env python
# coding: utf-8

import requests
import json
import numpy as np

BASE_URL = 'http://127.0.0.1:8080'

def test_endpoint(name, method, url, data=None):
    """Función genérica para probar un endpoint."""
    print(f"--- Probando: {name} ---")
    print(f"Petición: {method.upper()} {url}")
    if data:
        # Mostrar solo un resumen si los datos son muy largos
        if 'features' in data:
            if isinstance(data['features'], list) and len(data['features']) > 0:
                if isinstance(data['features'][0], list):
                    print(f"Enviando {len(data['features'])} muestras, cada una con {len(data['features'][0])} características")
                else:
                    print(f"Enviando 1 muestra con {len(data['features'])} características")
    
    try:
        if method == 'get':
            response = requests.get(url)
        elif method == 'post':
            response = requests.post(url, json=data)
        else:
            print("Método HTTP no soportado.")
            return

        print(f"Código de estado: {response.status_code}")
        try:
            print(f"Respuesta del servidor: {json.dumps(response.json(), indent=2)}")
        except json.JSONDecodeError:
            print(f"Respuesta del servidor (no es JSON): {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"No se pudo conectar al API: {e}")
    
    print("-" * 50 + "\n")

# --- Datos de ejemplo reales del dataset Breast Cancer ---

# Ejemplo de muestra maligna (clase 0)
malignant_sample = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 
                   0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 
                   0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 
                   0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]

# Ejemplo de muestra benigna (clase 1)
benign_sample = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 
                0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 
                0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 
                0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]


# In[7]:


# 1. Prueba de Estado del Servicio
test_endpoint(
    name="Estado del Servicio (/health)",
    method='get',
    url=f"{BASE_URL}/health")


# In[8]:


# Prueba de Información del Modelo
test_endpoint(
    name="Información del Modelo (/model-info)",
    method='get',
    url=f"{BASE_URL}/model-info")


# - **Prueba de predicción de casos**

# In[9]:


# Prueba de Predicción Individual - Muestra Maligna
test_endpoint(
    name="Predicción Individual - Muestra Maligna",
    method='post',
    url=f"{BASE_URL}/predict",
    data={"features": malignant_sample}
)

# Prueba de Predicción Individual - Muestra Benigna
test_endpoint(
    name="Predicción Individual - Muestra Benigna",
    method='post',
    url=f"{BASE_URL}/predict",
    data={"features": benign_sample}
)

# Prueba de Predicción por Lote
test_endpoint(
    name="Predicción por Lote (3 muestras)",
    method='post',
    url=f"{BASE_URL}/predict",
    data={
        "features": [
            malignant_sample,
            benign_sample,
            [13.08, 15.71, 85.63, 520.0, 0.1075, 0.127, 0.04568, 0.0311, 
             0.1967, 0.06811, 0.1852, 0.7477, 1.383, 14.67, 0.004097, 0.01898, 
             0.01698, 0.00649, 0.01678, 0.002425, 14.5, 20.49, 96.09, 630.5, 
             0.1312, 0.2776, 0.189, 0.07283, 0.3184, 0.08183]
        ]
    }
)


# - **Prueba de errores**

# In[16]:


# Prueba de Error: Clave 'features' ausente en el JSON
test_endpoint(
    name="Error de Petición (Clave 'features' ausente)",
    method='post',
    url=f"{BASE_URL}/predict",
    data={
        "datos": [benign_sample]})


# In[17]:


# Prueba de Error: Menos características de las esperadas
test_endpoint(
    name="Error de Predicción (Solo 20 características)",
    method='post',
    url=f"{BASE_URL}/predict",
    data={
        "features": malignant_sample[:20]  # <-- Solo 20 características en lugar de 30
    }
)


# In[20]:


# Prueba de Error: JSON vacío
test_endpoint(
    name="Error de Petición (JSON vacío)",
    method='post',
    url=f"{BASE_URL}/predict",
    data={}  # <-- JSON vacío
)


# In[ ]:


# Convertir a script
get_ipython().system('jupyter nbconvert --to script pruebas.ipynb')

