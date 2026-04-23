import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, max_error

from VotingBagging import ensemble


def compare_single_car(car_dict, actual_price=None, csv_file="Data/parsedData.csv"):
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

def compare_top_5000_cars(csv_file="Data/parsedData.csv", top_n=5000):
    df = pd.read_csv(csv_file).copy()

    # Adjust this if your target column has a different name
    target_col = "price"

    df = df.dropna(subset=[target_col])

    train_df = df.iloc[top_n:60000].copy()
    valid_df = df.iloc[:top_n].copy()

    train_df = ensemble.clean_car_dataframe(train_df)
    valid_df = ensemble.clean_car_dataframe(valid_df)

    models = ensemble.load_or_train_all_models(train_df)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    X_valid = valid_df.drop(columns=[target_col])
    y_valid = valid_df[target_col].values

    train_pred_log = ensemble.ensemble_predict_log(X_train, models)
    valid_pred_log = ensemble.ensemble_predict_log(X_valid, models)

    train_pred = np.expm1(train_pred_log)
    valid_pred = np.expm1(valid_pred_log)

    results = {
        "train_mae": mean_absolute_error(y_train, train_pred),
        "valid_mae": mean_absolute_error(y_valid, valid_pred),
        "train_median_ae": median_absolute_error(y_train, train_pred),
        "valid_median_ae": median_absolute_error(y_valid, valid_pred),
        "train_max_ae": max_error(y_train, train_pred),
        "valid_max_ae": max_error(y_valid, valid_pred),
    }

    print("Training metrics:")
    print(f"MAE:     ${results['train_mae']:,.2f}")
    print(f"Median:  ${results['train_median_ae']:,.2f}")
    print(f"MAX AE:  ${results['train_max_ae']:,.2f}")

    print("\nValidation metrics:")
    print(f"MAE:     ${results['valid_mae']:,.2f}")
    print(f"Median:  ${results['valid_median_ae']:,.2f}")
    print(f"MAX AE:  ${results['valid_max_ae']:,.2f}")

    return results


def main():
    csv_file = "Data/parsedData.csv"

    results = compare_top_5000_cars(csv_file=csv_file, top_n=5000)
    print("Complete!")


if __name__ == "__main__":
    main()