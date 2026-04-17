import numpy as np
import pandas as pd

from VotingBagging import ensemble


def compare_single_car(car_dict, actual_price=None, csv_file="Data/parsedData.csv"):
    # Load all models (RF, SVM, ANN, KNN)
    print("Loading or training all models...")
    models = ensemble.load_or_train_all_models(csv_file)

    # Convert input car to DataFrame
    car_df = pd.DataFrame([car_dict])
    car_df = ensemble.clean_car_dataframe(car_df)

    # Individual model predictions (log scale)
    predictions_log = {
        name: model.predict(car_df)[0]
        for name, model in models.items()
    }

    # Convert to dollar prices
    predictions_price = {
        name: np.expm1(pred)
        for name, pred in predictions_log.items()
    }

    # Ensemble prediction
    ensemble_log = ensemble.ensemble_predict_log(car_df, models)[0]
    ensemble_price = np.expm1(ensemble_log)

    # -------------------------
    # Print Results
    # -------------------------
    print("=== Individual Model Predictions ===")
    for name, price in predictions_price.items():
        print(f"{name.upper():<5}: ${price:,.2f}")

    print("\n=== Ensemble Prediction ===")
    print(f"Final Predicted Price: ${ensemble_price:,.2f}")

    # -------------------------
    # Compare to Actual Price
    # -------------------------
    if actual_price is not None:
        print(f"\nActual Price: ${actual_price:,.2f}")

        print("\n=== Errors ===")
        for name, price in predictions_price.items():
            error = abs(price - actual_price)
            print(f"{name.upper():<5}: ${error:,.2f}")

        ensemble_error = abs(ensemble_price - actual_price)
        print(f"ENSEM: ${ensemble_error:,.2f}")

    return ensemble_price


def main():
    car = {
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

    compare_single_car(car, actual_price=12900)


if __name__ == "__main__":
    main()