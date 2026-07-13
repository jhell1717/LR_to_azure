import os
import joblib
import numpy as np
import onnxruntime as rt
from onnx_utils import to_onnx_input

from data import load_raw, get_X_y, FEATURE_COLS, CATEGORICAL_COLS

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "model.joblib"
)
_DEFAULT_ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "model.onnx"
)


def main():
    df = load_raw()
    X, y = get_X_y(df)
    X_sample = X.sample(200,random_state=1)

    sk_pipeline = joblib.load(_DEFAULT_MODEL_PATH)
    sk_probs = sk_pipeline.predict_proba(X_sample)[:,1]

    sess = rt.InferenceSession(_DEFAULT_ONNX_PATH)
    onnx_inputs = to_onnx_input(X_sample)
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(output_names, onnx_inputs)

    onnx_probs = results[1][:,1]

    max_diff = np.abs(sk_probs - onnx_probs).max()

    print("Max probability difference:", max_diff)
    assert max_diff < 1e-4, "ONNX output diverges from sklearn output!"
    print("✅ ONNX model matches sklearn model")

if __name__ == "__main__":
    main()