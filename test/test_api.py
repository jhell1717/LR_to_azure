# tests/test_api.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_predict_valid_input():
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert "churn_prediction" in body
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0

def test_predict_rejects_bad_input():
    bad_customer = dict(SAMPLE_CUSTOMER)
    del bad_customer["tenure"]  # missing required field
    resp = client.post("/predict", json=bad_customer)
    assert resp.status_code == 422  # FastAPI validation error
