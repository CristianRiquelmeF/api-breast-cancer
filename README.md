# API de Predicción de Cáncer de Mama

Esta es una API RESTful desarrollada en Python con Flask que utiliza un modelo de Machine Learning para predecir si un tumor de cáncer de mama es benigno o maligno.

Descripción del Proyecto 

El objetivo de este proyecto es proporcionar una interfaz de programación de aplicaciones (API) que permita a los usuarios obtener predicciones sobre el diagnóstico de cáncer de mama. La API recibe como entrada un conjunto de características de un tumor y devuelve una predicción de si el tumor es benigno o maligno, basándose en un modelo de Máquinas de Vectores de Soporte (SVM) previamente entrenado.

## Características

- **Modelo de ML**: Random Forest Classifier optimizado con GridSearchCV
- **API REST**: Desarrollada con FastAPI
- **Documentación Automática**: Swagger UI y ReDoc integrados
- **Validación de Datos**: Usando Pydantic
- **Preprocesamiento**: Escalado de características y encoding
- **Métricas**: Evaluación completa con curvas ROC y reportes de clasificación

## Tecnologías Utilizadas

- **Backend**: FastAPI, Uvicorn
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn
- **Serialización**: Joblib
- **Type Hints**: Pydantic

  ---

# Modelo de Machine Learning
#### Dataset

    Fuente: Wisconsin Breast Cancer Dataset

    Características: 30 features numéricos

    Target: 0 (Benigno), 1 (Maligno)

#### Entrenamiento

El modelo fue entrenado con las siguientes especificaciones:

    Algoritmo: Random Forest Classifier

    Optimización: GridSearchCV con validación cruzada

    Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC

    Rendimiento: AUC-ROC de 0.99 (Excelente)

  ---

Instalación 

Para ejecutar este proyecto localmente, sigue estos pasos:

    Clona el repositorio: git clone https://github.com/CristianRiquelmeF/api-breast-cancer.git

---
> Proyecto desarrollado como parte de portafolio de Machine Learning avanzado.
---

#### Diclaimer
> Este proyecto es para fines educativos y de demostración. No debe ser utilizado para diagnósticos médicos reales. Siempre consulte con profesionales de la salud para diagnósticos médicos.
