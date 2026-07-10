# src/train.py
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import os

from data import load_raw, get_X_y, CATEGORICAL_COLS, NUMERIC_COLS


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "model.joblib"
)

def main():
    df = load_raw()
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    os.makedirs(os.path.dirname(_DEFAULT_MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, _DEFAULT_MODEL_PATH)
    print("Saved to models/model.joblib")


if __name__ == "__main__":
    main()