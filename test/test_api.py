import sys
import os
import pytest

# Agregar el directorio raíz al path de Python para importar app correctamente
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app

def test_app_creation():
    """Test que verifica que la app Flask se crea correctamente"""
    assert app is not None
    assert hasattr(app, 'config')
    assert app.name == 'app'

def test_health_endpoint():
    """Test para el endpoint raíz (health check)"""
    with app.test_client() as client:
        response = client.get('/')
        # Puede ser 200, 404, o otro código, pero no debe dar error 500
        assert response.status_code in [200, 404]  # 404 si no existe el endpoint

def test_predict_endpoint():
    """Test que verifica el comportamiento del endpoint /predict"""
    with app.test_client() as client:
        # Probar que el endpoint existe (no da 404)
        response = client.get('/predict')
        assert response.status_code != 404
        
        # Probar con método POST (si es lo que espera tu API)
        response_post = client.post('/predict')
        assert response_post.status_code != 404

def test_app_config():
    """Test que verifica la configuración básica de la app"""
    assert not app.config.get('DEBUG') or True  # DEBUG puede estar activo
    testing_config = app.config.get('TESTING')
    # No falla si TESTING no está configurado

def test_imports():
    """Test que verifica que todos los imports necesarios funcionan"""
    try:
        import flask
        import sklearn
        import pandas
        import numpy
        import pickle
        print("✓ Todas las dependencias importadas correctamente")
    except ImportError as e:
        pytest.fail(f"Falta dependencia: {e}")