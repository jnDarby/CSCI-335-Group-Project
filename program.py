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

def compare_top_5000_cars(car_df, csv_file="parsed_data.csv", top_n=5000):
    df = pd.read_csv(csv_file)

    # keep only rows with a known target if needed
    # change 'price' to your actual target column name
    df = df.dropna(subset=["price"]).copy()

    # top 5000 rows for testing, rest for training
    test_df = df.head(top_n).copy()
    train_df = df.iloc[top_n:].copy()

    train_df = ensemble.clean_car_dataframe(train_df)
    test_df = ensemble.clean_car_dataframe(test_df)

    models = ensemble.load_or_train_all_models(csv_file)

    # evaluate on the top 5000
    y_true = test_df["price"].values
    X_test = test_df.drop(columns=["price"])

    pred_log = ensemble.ensemble_predict_log(X_test, models)
    preds = np.expm1(pred_log)

    abs_errors = np.abs(preds - y_true)

    print(f"Evaluated on top {len(test_df)} cars")
    print(f"MAE: ${abs_errors.mean():,.2f}")
    print(f"Median AE: ${np.median(abs_errors):,.2f}")
    print(f"Max AE: ${abs_errors.max():,.2f}")

    return {
        "mae": abs_errors.mean(),
        "median_ae": np.median(abs_errors),
        "max_ae": abs_errors.max(),
        "predictions": preds,
        "actuals": y_true
    }


def main():
    csv_file = "parsed_data.csv"

    results = compare_top_5000_cars(csv_file=csv_file, top_n=5000)

    print("\nFinal results:")
    print(f"MAE: ${results['mae']:,.2f}")
    print(f"Median AE: ${results['median_ae']:,.2f}")
    print(f"Max AE: ${results['max_ae']:,.2f}")


if __name__ == "__main__":
    main()