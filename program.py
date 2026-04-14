import numpy as np
import pandas as pd

from VotingBagging import ensemble


def compare_single_car(car_dict, actual_price=None, csv_file="parsed_data.csv"):
    models = ensemble.load_or_train_all_models(csv_file)

    car_df = pd.DataFrame([car_dict])
    car_df = ensemble.clean_car_dataframe(car_df)

    pred_log = ensemble.ensemble_predict_log(car_df, models)[0]
    predicted_price = np.expm1(pred_log)

    print(f"Predicted price: ${predicted_price:,.2f}")

    if actual_price is not None:
        difference = predicted_price - actual_price
        abs_error = abs(difference)

        print(f"Actual price:    ${actual_price:,.2f}")
        print(f"Difference:      ${difference:,.2f}")
        print(f"Absolute error:  ${abs_error:,.2f}")

    return predicted_price


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