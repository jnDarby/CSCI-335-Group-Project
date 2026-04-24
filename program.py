import numpy as np
import pandas as pd

from VotingBagging import ensemble
import knn

def evaluate_knn(csv_file="Data/parsedData.csv", output_csv="knn_results.csv"):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    start_time = time.time()

    print("1. Loading and cleaning dataset...")
    data = knn.load_and_clean_data(csv_file)
    print(f"   Rows loaded: {len(data)}")

    print("2. Preparing features and target...")
    X = data.drop(columns=["price", "id"])
    y = np.log1p(data["price"])

    print("3. Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"   Train rows: {len(X_train)}")
    print(f"   Test rows:  {len(X_test)}")

    print("4. Building model...")
    model = knn.build_pipeline()

    print("5. Training...")
    model.fit(X_train, y_train)

    print("6. Predicting...")
    pred_log = model.predict(X_test)

    print("7. Calculating metrics...")
    actual = np.expm1(y_test)
    predicted = np.expm1(pred_log)

    error = predicted - actual
    abs_error = np.abs(error)
    pct_error = (abs_error / actual) * 100

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    # Build export dataframe
    results = X_test.copy().reset_index(drop=True)
    results["ActualPrice"] = actual.reset_index(drop=True)
    results["PredictedPrice"] = predicted
    results["Error"] = error
    results["AbsoluteError"] = abs_error
    results["PercentError"] = pct_error

    # Save detailed results
    results.to_csv(output_csv, index=False)

    # Save metrics summary
    summary = pd.DataFrame([
        ["MAE", mae],
        ["RMSE", rmse],
        ["R2", r2],
        ["RowsTested", len(results)],
        ["Seconds", round(time.time() - start_time, 2)]
    ], columns=["Metric", "Value"])

    summary.to_csv("knn_summary.csv", index=False)

    print("\n=== KNN Results ===")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:   {r2:.4f}")
    print(f"Saved detailed results to {output_csv}")
    print("Saved summary to knn_summary.csv")

    return results

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

    #compare_single_car(car, actual_price=12900)
    evaluate_knn()


if __name__ == "__main__":
    main()