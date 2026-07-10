from data import load_raw, get_X_y, FEATURE_COLS

def test_data_loads():
    df = load_raw()
    assert len(df) > 7000
    assert "Churn" in df.columns

def test_get_X_y_shapes():
    df = load_raw()
    X, y = get_X_y(df)
    assert list(X.columns) == FEATURE_COLS
    assert y.isin([0, 1]).all()
    assert X.isnull().sum().sum() == 0  # TotalCharges NaNs should be filled

def test_no_leakage():
    df = load_raw()
    X, y = get_X_y(df)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns