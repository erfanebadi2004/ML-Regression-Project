import pytest
from fastapi.testclient import TestClient
from api_model import app

client = TestClient(app)

@pytest.fixture
def valid_sample_input():

    return {"samples": [
        {"store_nbr": 1, "family": "AUTOMOTIVE", "onpromotion": 0, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13,
        "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},

        {"store_nbr": 1, "family": "BABY CARE", "onpromotion": 0, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13,
         "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},

        {"store_nbr": 1, "family": "BEAUTY", "onpromotion": 1, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13,
         "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},

        {"store_nbr": 1, "family": "BEVERAGES", "onpromotion": 38, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13,
         "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},

        {"store_nbr": 1, "family": "BOOKS", "onpromotion": 0, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13,
         "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},
    ]}

def test_health_check():

    response = client.get("/health")
    data = response.json()

    assert data["status_code"] == 200
    assert data["status"] == "Healthy"
    assert data["model_loaded"] == True

def test_prediction_valid_input(valid_sample_input):

    response = client.post("/predict", json=valid_sample_input)

    assert response.status_code == 200

    result_response = response.json()

    assert "prediction result" in result_response

    assert len(result_response["prediction result"]) == len(valid_sample_input["samples"])

def test_prediction_invalid_input():

    sample_payload = {
        "InvalidField": [
            {"store_nbr": 1, "family": "AUTOMOTIVE", "onpromotion": 0, "city": "Quito", "state": "Pichincha",
             "type": "D", "cluster": 13,
             "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},

            {"store_nbr": 1, "family": "BABY CARE", "onpromotion": 0, "city": "Quito", "state": "Pichincha",
             "type": "D", "cluster": 13,
             "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},

            {"store_nbr": 1, "family": "BEAUTY", "onpromotion": 1, "city": "Quito", "state": "Pichincha", "type": "D",
             "cluster": 13,
             "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},

            {"store_nbr": 1, "family": "BEVERAGES", "onpromotion": 38, "city": "Quito", "state": "Pichincha",
             "type": "D", "cluster": 13,
             "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},

            {"store_nbr": 1, "family": "BOOKS", "onpromotion": 0, "city": "Quito", "state": "Pichincha", "type": "D",
             "cluster": 13,
             "holiday": 0, "year": 2017, "month": 7, "day": 15, "day_of_week": 5, "is_weekend": 0, "is_month_end": 1},
        ]
    }

    invalid_input = client.post(url="/predict", json=sample_payload)

    assert invalid_input.status_code == 422
