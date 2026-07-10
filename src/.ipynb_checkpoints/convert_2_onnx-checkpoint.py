import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from data import FEATURE_COLS, CATEGORICAL_COLS

def build_initial_types():
    initial_types = []
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            initial_types.alsppend((col, StringTensorType([None, 1])))
        else:
            initial_types.append((col, FloatTensorType([None, 1])))
    return initial_types

_DEFAULT_ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "model.onnx"
)

def main():
    pipeline = joblib.load("/Users/joshuahellewell/Desktop/01-dev/ml_deploy/src/models/model.joblib")
    initial_types = build_initial_types()

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        options={id(pipeline):{"zipmap": False}},
    )
    os.makedirs(os.path.dirname(_DEFAULT_ONNX_PATH), exist_ok=True)

    with open(_DEFAULT_ONNX_PATH,"wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Saved models/model.onnx")

if __name__=="__main__":
    main()
    