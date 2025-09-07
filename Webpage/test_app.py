import pytest
from flask import template_rendered
from app import create_app, validate_inputs, Converter, Weather, Config

from unittest.mock import patch, MagicMock
import pandas as pd

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test main index route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"html" in response.data or b"HTML" in response.data  # crude check content is HTML

def test_terms_and_conditions_route(client):
    response = client.get('/terms-and-conditions')
    assert response.status_code == 200

def test_data_visualization_route(client):
    response = client.get('/data-visualization')
    assert response.status_code == 200

def test_model_showcase_route(client):
    response = client.get('/model-showcase')
    assert response.status_code == 200

def test_404(client):
    response = client.get('/some-nonexistent-route')
    assert response.status_code == 404

def test_validate_inputs_valid():
    assert validate_inputs("01-01-2022", "5") == True

def test_validate_inputs_invalid_date():
    assert not validate_inputs("2022-01-01", "5")   # Wrong format

def test_validate_inputs_invalid_location():
    assert not validate_inputs("01-01-2022", "abc") # Not a digit
    assert not validate_inputs("01-01-2022", "-1")
    assert not validate_inputs("01-01-2022", "20")  # Out of allowed range

def test_converter_seconds_to_hours_minutes():
    assert Converter.convert_seconds_to_hours_minutes(3661) == "01:01"
    assert Converter.convert_seconds_to_hours_minutes(None) == "00:00"

def test_converter_date_to_string():
    result = Converter.convert_date_to_string("04-09-2025")
    assert "Thursday" in result and "September" in result and "2025" in result

@patch('Webpage.app.joblib.load')
def test_weather_model_loading(mock_joblib):
    # Mock label encoder and models
    mock_encoder = MagicMock()
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_joblib.side_effect = [mock_encoder] + [mock_model]*len(Config.TARGET_FEATURES)

    weather = Weather(Config)
    weather.load_resources(Config)
    assert weather.label_encoder == mock_encoder
    assert all(m == mock_model for m in weather.models.values())

@patch('app.joblib.load')
def test_weather_week_forecast(mock_joblib):
    # Setup mock encoder and models
    mock_encoder = MagicMock()
    mock_model = MagicMock()
    mock_model.predict.return_value = [10]  # Always predict 10 for any target
    mock_joblib.side_effect = [mock_encoder] + [mock_model]*len(Config.TARGET_FEATURES)

    weather = Weather(Config)
    weather.load_resources(Config)
    preds, warnings = weather.week_forecast("01-01-2022", 1)
    assert len(preds) == 7
    assert isinstance(warnings, list)
    assert "date" in preds[0]
    assert "precip" in preds[0]

def test_weather_warning_check():
    d = {"precip": 25, "feelslikemax c": 42, "windspeed": 25}
    warnings = Weather.warning_check(d)
    # Should contain severe warnings for all three
    assert any("Severe" in w for w in warnings)

def test_weather_descriptions():
    pdict = {"feelslikemax": 36, "humidity": 80, "cloudcover": 80, "precipcover": 80, "precip": 12, "windspeed": 35}
    short, long, bg = Weather.descriptions(pdict)
    assert isinstance(short, str)
    assert isinstance(long, str)
    assert isinstance(bg, str)