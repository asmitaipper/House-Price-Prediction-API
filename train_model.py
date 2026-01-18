
End‑to‑end FastAPI + Docker ML projects follow a very similar README structure.[2][1]

***

## train_model.py (training script)

This is a simple, resume‑ready training script using a CSV with columns like `rooms,area,age,location_score,price`. You can adapt names to your dataset.[5][2]

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
import os

DATA_PATH = "data/housing.csv"
MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.joblib")

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def prepare_xy(df: pd.DataFrame):
    # Adjust column names according to your CSV
    target_col = "price"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return X, y, numeric_cols, categorical_cols

def build_pipeline(numeric_cols, categorical_cols):
    from sklearn.preprocessing import OneHotEncoder

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipe

def main():
    df = load_data(DATA_PATH)
    X, y, num_cols, cat_cols = prepare_xy(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(num_cols, cat_cols)

    print("Training model...")
    pipe.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
