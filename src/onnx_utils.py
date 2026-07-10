# src/onnx_utils.py
import numpy as np
from data import FEATURE_COLS, CATEGORICAL_COLS

def to_onnx_input(X):
    """Convert a pandas DataFrame into the dict-of-numpy-arrays format
    onnxruntime expects. Uses np.asarray(..., dtype=...) rather than
    .values.astype(...) because pandas' nullable 'string' dtype does not
    convert cleanly to a plain numpy array via .astype alone."""
    inputs = {}
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            inputs[col] = np.asarray(X[col], dtype=str).reshape(-1, 1)
        else:
            inputs[col] = np.asarray(X[col], dtype=np.float32).reshape(-1, 1)
    return inputs