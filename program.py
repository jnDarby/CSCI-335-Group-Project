import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    max_error,
    mean_squared_error,
    r2_score,
)

from VotingBagging import ensemble


def prepare_dataframe(df, target_col="price"):
    df = df.copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Basic outlier filtering
    df = df[df[target_col].between(500, 100000)]

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].between(1990, 2026)]

    if "odometer" in df.columns:
        df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
        df = df[df["odometer"].between(0, 400000)]

    df = ensemble.clean_car_dataframe(df)
    return df


def compare_single_car(car_dict, actual_price=None, csv_file="Data/parsedData.csv"):
    df = pd.read_csv(csv_file)
    df = prepare_dataframe(df)

    models = ensemble.load_or_train_all_models(df)

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


def evaluate_predictions(y_true, y_pred):
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "max_ae": max_error(y_true, y_pred),
        "p90_ae": np.percentile(abs_errors, 90),
        "p95_ae": np.percentile(abs_errors, 95),
        "bias": np.mean(errors),
    }


def compare_top_5000_cars(csv_file="Data/parsedData.csv", top_n=5000):
    df = pd.read_csv(csv_file)
    df = prepare_dataframe(df)

    target_col = "price"

    valid_df = df.iloc[:top_n].copy()
    train_df = df.iloc[top_n:60000].copy()

    models = ensemble.load_or_train_all_models(train_df)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    X_valid = valid_df.drop(columns=[target_col])
    y_valid = valid_df[target_col].values

    train_pred_log = ensemble.ensemble_predict_log(X_train, models)
    valid_pred_log = ensemble.ensemble_predict_log(X_valid, models)

    train_pred = np.expm1(train_pred_log)
    valid_pred = np.expm1(valid_pred_log)

    train_results = evaluate_predictions(y_train, train_pred)
    valid_results = evaluate_predictions(y_valid, valid_pred)

    print("Training metrics:")
    print(f"MAE:      ${train_results['mae']:,.2f}")
    print(f"Median:   ${train_results['median_ae']:,.2f}")
    print(f"RMSE:     ${train_results['rmse']:,.2f}")
    print(f"R^2:      {train_results['r2']:.4f}")
    print(f"P90 AE:   ${train_results['p90_ae']:,.2f}")
    print(f"P95 AE:   ${train_results['p95_ae']:,.2f}")
    print(f"Bias:     ${train_results['bias']:,.2f}")
    print(f"MAX AE:   ${train_results['max_ae']:,.2f}")

    print("\nValidation metrics:")
    print(f"MAE:      ${valid_results['mae']:,.2f}")
    print(f"Median:   ${valid_results['median_ae']:,.2f}")
    print(f"RMSE:     ${valid_results['rmse']:,.2f}")
    print(f"R^2:      {valid_results['r2']:.4f}")
    print(f"P90 AE:   ${valid_results['p90_ae']:,.2f}")
    print(f"P95 AE:   ${valid_results['p95_ae']:,.2f}")
    print(f"Bias:     ${valid_results['bias']:,.2f}")
    print(f"MAX AE:   ${valid_results['max_ae']:,.2f}")

    return {
        "train": train_results,
        "valid": valid_results,
    }


def main():
    csv_file = "Data/parsedData.csv"
    results = compare_top_5000_cars(csv_file=csv_file, top_n=5000)
    print("Complete!")
    return results


if __name__ == "__main__":
    main()