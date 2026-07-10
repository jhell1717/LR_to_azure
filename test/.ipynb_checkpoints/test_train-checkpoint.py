from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from data import load_raw, get_X_y
from train import build_pipeline

def test_pipeline_trains_and_beats_baseline():
    df = load_raw()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    # Regression guard: fail CI if a future change tanks model quality
    assert auc > 0.75, f"AUC {auc:.3f} dropped below acceptable threshold"