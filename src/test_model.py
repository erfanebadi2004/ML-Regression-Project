import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from api_model import app

client = TestClient(app)

@pytest.fixture
def sample_payload():
    prediction_test = pd.read_csv("Files/prediction_test.csv")

    data_list = prediction_test.to_dict(orient='records')

    return {"samples": data_list}

def test_prediction(sample_payload):
    response = client.post("/predict", json=sample_payload)

    result_response = response.json()

    print("\n--- Model Predictions ---")
    print(np.round(result_response, 0))
    print("===== ===== ===== ===== =====")

    assert response.status_code == 200



































#import numpy as np
#
#import requests

#prediction_test = pd.read_csv("Files/prediction_test.csv")

#data_list = prediction_test.to_dict(orient='records')

#payload = {
#    "samples": data_list
#}

#url = "http://127.0.0.1:8000/predict"

#response = requests.post(url, json=payload)

#print(response)
#result_prediction = response.json()

#prediction_test["result_prediction"] = np.round(result_prediction, 0)

#for row in prediction_test.itertuples():
#    list_row = list(row)
#    print(list_row)
#    print("_" * 100)