import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from randomForest import build_pipeline as build_rf_pipeline
from svm_model import build_pipeline as build_svm_pipeline
from ann_model import build_pipeline as build_ann_pipeline
from randomForest import load_and_clean_data


MODEL_FILE = "ensemble_used_car_price_model.joblib"


def build_pipeline():
    rf_pipeline = clone(build_rf_pipeline())
    svm_pipeline = clone(build_svm_pipeline())
    ann_pipeline = clone(build_ann_pipeline())

    ensemble = VotingRegressor(
        estimators=[
            ("rf", rf_pipeline),
            ("svm", svm_pipeline),
            ("ann", ann_pipeline)
        ],
        weights=[3, 2, 2],
        n_jobs=-1
    )

    return ensemble


def train_model(csv_file="parsed_data.csv"):
    data = load_and_clean_data(csv_file)
    print(f"Data loaded. Rows: {len(data)}")

    X = data.drop(columns=["price", "id"])
    y = np.log1p(data["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    predictions = np.expm1(pred_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5
    r2 = r2_score(actual, predictions)

    print(f"Ensemble MAE:  {mae:.2f}")
    print(f"Ensemble RMSE: {rmse:.2f}")
    print(f"Ensemble R^2:  {r2:.4f}")

    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    return model


def predict_price(car_dict, model_file=MODEL_FILE):
    model = joblib.load(model_file)
    car_df = pd.DataFrame([car_dict])

    for col in car_df.columns:
        if car_df[col].dtype == "object":
            car_df[col] = car_df[col].astype(str).str.strip().str.lower()

    pred_log = model.predict(car_df)[0]
    return np.expm1(pred_log)


def main():
    train_model("parsed_data.csv")

    example_car = {
        "year": 2015,
        "manufacturer": "toyota",
        "model": "camry",
        "condition": "good",
        "cylinders": "4 cylinders",
        "fuel": "gas",
        "odometer": 85000,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "silver"
    }

    estimated_price = predict_price(example_car)
    print(f"Estimated price: ${estimated_price:,.2f}")


if __name__ == "__main__":
    main()