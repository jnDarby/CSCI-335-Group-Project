import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_FILE = "used_car_price_model.joblib"

RELEVANT_COLUMNS = [
    "id", "price", "year", "manufacturer", "model", "condition",
    "cylinders", "fuel", "odometer", "title_status",
    "transmission", "drive", "type", "paint_color"
]

TARGET = "price"


def load_and_clean_data(csv_file):
    data = pd.read_csv(csv_file, usecols=RELEVANT_COLUMNS)

    data = data.dropna(subset=[TARGET, "year", "manufacturer", "model", "odometer"])
    data = data.sample(min(len(data), 10000), random_state=42)

    data = data[data["price"] > 500]
    data = data[data["price"] < 100000]
    data = data[data["year"].between(1990, 2026)]
    data = data[data["odometer"].between(0, 400000)]

    for col in ["manufacturer", "model", "condition", "cylinders", "fuel",
                "title_status", "transmission", "drive", "type", "paint_color"]:
        data[col] = data[col].astype(str).str.strip().str.lower()

    return data


def build_pipeline():
    numeric_features = ["year", "odometer"]
    categorical_features = [
        "manufacturer", "model", "condition", "cylinders", "fuel",
        "title_status", "transmission", "drive", "type", "paint_color"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=20))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def train_model(csv_file="parsed_data.csv"):
    data = load_and_clean_data(csv_file)

    X = data.drop(columns=["price", "id"])
    y = np.log1p(data["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()

    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [10, 20, 30, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    pred_log = best_model.predict(X_test)
    predictions = np.expm1(pred_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5
    r2 = r2_score(actual, predictions)

    print(f"Best params: {search.best_params_}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2:  {r2:.4f}")

    if hasattr(best_model.named_steps["model"], "oob_score_"):
        print(f"OOB R^2: {best_model.named_steps['model'].oob_score_:.4f}")

    joblib.dump(best_model, MODEL_FILE)
    return best_model


def predict_price(car_dict, model_file=MODEL_FILE):
    model = joblib.load(model_file)
    car_df = pd.DataFrame([car_dict])

    for col in car_df.columns:
        if car_df[col].dtype == "object":
            car_df[col] = car_df[col].astype(str).str.strip().str.lower()

    predicted_log_price = model.predict(car_df)[0]
    return np.expm1(predicted_log_price)


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
    print(f"Estimated price for example car: ${estimated_price:,.2f}")


if __name__ == "__main__":
    main()