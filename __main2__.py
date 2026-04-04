import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_FILE = "used_car_price_model.joblib"


RELEVANT_COLUMNS = [
    'id', 'price', 'year', 'manufacturer', 'model', 'condition',
    'cylinders', 'fuel', 'odometer', 'title_status',
    'transmission', 'drive', 'type', 'paint_color'
]


TARGET = "price"


def load_and_clean_data(csv_file):
    data = pd.read_csv(csv_file, usecols=RELEVANT_COLUMNS)

    data = data.dropna(subset=[TARGET, "year", "manufacturer", "model", "odometer"])
    data = data.head(10000) # limited data to the first 10,000 rows, optional for speed testing

    data = data[data["price"] > 0]
    data = data[data["year"] > 1900]
    data = data[data["odometer"] >= 0]

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
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def train_model(csv_file="parsed_data.csv"):
    data = load_and_clean_data(csv_file)
    print(f"Data loaded and cleaned. Rows: {len(data)}")

    X = data.drop(columns=["price", "id"])
    y = data["price"]
    print(f"Features and target separated. X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training and testing sets created. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("Model training completed.")

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"Rows used: {len(data)}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2:  {r2:.4f}")

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    return pipeline


def predict_price(car_dict, model_file=MODEL_FILE):
    model = joblib.load(model_file)
    car_df = pd.DataFrame([car_dict])
    predicted_price = model.predict(car_df)[0]
    return predicted_price


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