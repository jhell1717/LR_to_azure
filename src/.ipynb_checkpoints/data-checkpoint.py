import pandas as pd
TARGET = "Churn"
ID_COL = "customerID"

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

def load_raw(path: str="/Users/joshuahellewell/Desktop/01-dev/ml_deploy/data/telco.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df

def get_X_y(df: pd.DataFrame):
    df = clean(df)
    X = df[FEATURE_COLS]
    y = df[TARGET]
    return X,y