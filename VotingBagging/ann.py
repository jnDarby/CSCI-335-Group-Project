import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_FILE = "ann_used_car_price_model.joblib"

RELEVANT_COLUMNS = [
    "id", "price", "year", "manufacturer", "model", "condition",
    "cylinders", "fuel", "odometer", "title_status",
    "transmission", "drive", "type", "paint_color"
]

TARGET = "price"


def load_and_clean_data(csv_file):
    data = pd.read_csv(csv_file, usecols=RELEVANT_COLUMNS)

    data = data.dropna(subset=[TARGET, "year", "manufacturer", "model", "odometer"])

    data = data[data["price"].between(500, 100000)]
    data = data[data["year"].between(1990, 2026)]
    data = data[data["odometer"].between(0, 400000)]

    categorical_cols = [
        "manufacturer", "model", "condition", "cylinders", "fuel",
        "title_status", "transmission", "drive", "type", "paint_color"
    ]

    for col in categorical_cols:
        data[col] = data[col].astype(str).str.strip().str.lower()

    return data


def build_pipeline():
    numeric_features = ["year", "odometer"]
    categorical_features = [
        "manufacturer", "model", "condition", "cylinders", "fuel",
        "title_status", "transmission", "drive", "type", "paint_color"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def train_model(csv_file="Data/parsedData.csv"):
    data = load_and_clean_data(csv_file)
    print(f"Data loaded. Rows: {len(data)}")

    X = data.drop(columns=["price", "id"])
    y = np.log1p(data["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_test)
    predictions = np.expm1(pred_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5
    r2 = r2_score(actual, predictions)

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2:  {r2:.4f}")

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    return pipeline


def predict_price(car_dict, model_file=MODEL_FILE):
    model = joblib.load(model_file)
    car_df = pd.DataFrame([car_dict])

    for col in car_df.columns:
        if car_df[col].dtype == "object":
            car_df[col] = car_df[col].astype(str).str.strip().str.lower()

    pred_log = model.predict(car_df)[0]
    return np.expm1(pred_log)