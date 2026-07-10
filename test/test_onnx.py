# tests/test_onnx.py
import os
import joblib
import numpy as np
import onnxruntime as rt
import pytest

from data import load_raw, get_X_y
from onnx_utils import to_onnx_input

MODEL_PATH = "/Users/joshuahellewell/Desktop/01-dev/ml_deploy/models/model.joblib"
ONNX_PATH = "/Users/joshuahellewell/Desktop/01-dev/ml_deploy/models/model.onnx"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.exists(ONNX_PATH)),
    reason="Trained model / ONNX file not present — run train.py and convert_to_onnx.py first",
)

def test_onnx_matches_sklearn():
    df = load_raw()
    X, y = get_X_y(df)
    X_sample = X.sample(100, random_state=1)

    sk_pipeline = joblib.load(MODEL_PATH)
    sk_probs = sk_pipeline.predict_proba(X_sample)[:, 1]

    sess = rt.InferenceSession(ONNX_PATH)
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(output_names, to_onnx_input(X_sample))
    onnx_probs = results[1][:, 1]

    max_diff = np.abs(sk_probs - onnx_probs).max()
    assert max_diff < 1e-4