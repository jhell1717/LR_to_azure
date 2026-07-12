import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI
from pydantic import BaseModel

from data import FEATURE_COLS, CATEGORICAL_COLS
from onnx_utils import to_onnx_input
import pandas as pd
import logging


from azure.monitor.opentelemetry import configure_azure_monitor

if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.onnx")

app = FastAPI(title="Churn Prediction API")

session = rt.InferenceSession(MODEL_PATH)

output_names = [o.name for o in session.get_outputs()]

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    return {"status":"ok"}

# @app.post("/predict")
# def predict(features: CustomerFeatures):
#     row = pd.DataFrame([features.model_dump()])[FEATURE_COLS]
#     onnx_inputs = to_onnx_input(row)
#     results = session.run(output_names, onnx_inputs)

#     label = int(results[0][0])
#     prob_churn = float(results[1][0][1])
#     return {
#         "churn_prediction" : bool(label),
#         "churn_probability" : round(prob_churn, 4),
#     }

logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(features: CustomerFeatures):
    row = pd.DataFrame([features.model_dump()])[FEATURE_COLS]
    onnx_inputs = to_onnx_input(row)
    results = session.run(output_names, onnx_inputs)

    label = int(results[0][0])
    prob_churn = float(results[1][0][1])

    logger.info(
        "prediction_made",
        extra={
            "custom_dimensions": {
                "churn_probability": prob_churn,
                "contract": features.Contract,
                "tenure": features.tenure,
                "monthly_charges": features.MonthlyCharges,
            }
        },
    )

    return {
        "churn_prediction": bool(label),
        "churn_probability": round(prob_churn, 4),
    }