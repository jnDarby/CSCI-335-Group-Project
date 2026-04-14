import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from .randomForest import train_model as train_rf_model
from .svm import train_model as train_svm_model
from .ann import train_model as train_ann_model

RF_MODEL_FILE = "random_forest_used_car_price_model.joblib"
SVM_MODEL_FILE = "svm_used_car_price_model.joblib"
ANN_MODEL_FILE = "ann_used_car_price_model.joblib"


def clean_car_dataframe(car_df):
    categorical_cols = [
        "manufacturer", "model", "condition", "cylinders", "fuel",
        "title_status", "transmission", "drive", "type", "paint_color"
    ]

    for col in categorical_cols:
        if col in car_df.columns:
            car_df[col] = car_df[col].astype(str).str.strip().str.lower()

    return car_df


def load_or_train_individual_model(model_file, train_function, csv_file="../Data/parsedData.csv"):
    model_path = Path(model_file)

    if model_path.exists():
        return joblib.load(model_file)

    return train_function(csv_file)


def load_or_train_all_models(csv_file="../Data/parsedData.csv"):
    rf_model = load_or_train_individual_model(RF_MODEL_FILE, train_rf_model, csv_file)
    svm_model = load_or_train_individual_model(SVM_MODEL_FILE, train_svm_model, csv_file)
    ann_model = load_or_train_individual_model(ANN_MODEL_FILE, train_ann_model, csv_file)

    return {
        "rf": rf_model,
        "svm": svm_model,
        "ann": ann_model
    }


def ensemble_predict_log(car_df, models, weights=None):
    if weights is None:
        weights = {"rf": 3, "svm": 2, "ann": 2}

    predictions = {
        name: model.predict(car_df)
        for name, model in models.items()
    }

    weighted_sum = sum(weights[name] * predictions[name] for name in predictions)
    total_weight = sum(weights.values())

    return weighted_sum / total_weight


if __name__ == "__main__":
    main()