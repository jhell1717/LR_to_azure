import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from data import FEATURE_COLS, CATEGORICAL_COLS

def build_initial_types():
    initial_types = []
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            initial_types.append((col, StringTensorType([None, 1])))
        else:
            initial_types.append((col, FloatTensorType([None, 1])))
    return initial_types

def main():
    pipeline = joblib.load("/Users/joshuahellewell/Desktop/01-dev/ml_deploy/src/models/model.joblib")
    initial_types = build_initial_types()

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        options={id(pipeline):{"zipmap": False}},
    )

    with open("/Users/joshuahellewell/Desktop/01-dev/ml_deploy/src/models/model.onnx","wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Saved models/model.onnx")

if __name__=="__main__":
    main()
    