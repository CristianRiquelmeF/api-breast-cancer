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

Instalación 

Para ejecutar este proyecto localmente, sigue estos pasos:

    Clona el repositorio:

git clone https://github.com/CristianRiquelmeF/api-breast-cancer.git

Crea un entorno virtual:

python -m venv env

Activa el entorno virtual:

    En Windows:

env\Scripts\activate

Instalación ⚙️

Para ejecutar este proyecto localmente, sigue estos pasos:

    Clona el repositorio:
    Bash

git clone https://github.com/CristianRiquelmeF/api-breast-cancer.git

Crea un entorno virtual:
Bash

python -m venv env

Activa el entorno virtual:

    En Windows:

env\Scripts\activate
    

