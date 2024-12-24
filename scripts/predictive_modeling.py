import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def tune_and_build_models(input_path, output_path):
    # Load the dataset
    df = pd.read_csv(input_path)

    # Prepare features and target
    X = df[['Temperature', 'Humidity']]
    y = df['GrowthRate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter grids
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }

    # Random Forest Tuning
    rf_model = RandomForestRegressor(random_state=42)
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    rf_grid_search.fit(X_train, y_train)
    rf_best_model = rf_grid_search.best_estimator_
    rf_predictions = rf_best_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")
    print(f"Tuned Random Forest MSE: {rf_mse:.4f}")

    # Gradient Boosting Tuning
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_grid_search = GridSearchCV(gb_model, gb_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    gb_grid_search.fit(X_train, y_train)
    gb_best_model = gb_grid_search.best_estimator_
    gb_predictions = gb_best_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_predictions)
    print(f"Best Gradient Boosting Parameters: {gb_grid_search.best_params_}")
    print(f"Tuned Gradient Boosting MSE: {gb_mse:.4f}")

    # Choose the best model
    if rf_mse < gb_mse:
        best_model = rf_best_model
        best_predictions = rf_predictions
        best_model_name = "Random Forest"
        best_mse = rf_mse
    else:
        best_model = gb_best_model
        best_predictions = gb_predictions
        best_model_name = "Gradient Boosting"
        best_mse = gb_mse

    print(f"Best Model: {best_model_name} with MSE: {best_mse:.4f}")

    # Save predictions to the dataset
    df['PredictedGrowthRate'] = best_model.predict(df[['Temperature', 'Humidity']])
    df.to_csv(output_path, index=False)
    print(f"Data with predictions saved to {output_path}")

    # Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_predictions, alpha=0.7, label="Predicted vs Actual")
    plt.plot([1.0, 2.0], [1.0, 2.0], color="red", linestyle="--", label="Ideal Fit")

    # Highlight outliers
    residuals = abs(best_predictions - y_test)
    outlier_threshold = 0.2
    outliers = residuals > outlier_threshold
    plt.scatter(y_test[outliers], best_predictions[outliers], color="red", label="Outliers")

    # Annotate MSE
    mse_text = f"{best_model_name} MSE: {best_mse:.4f}"
    plt.text(1.0, 1.9, mse_text, fontsize=10, color="black")

    # Add title and labels
    plt.title(f"Growth Rate Predictions ({best_model_name})")
    plt.xlabel("Actual Growth Rate")
    plt.ylabel("Predicted Growth Rate")
    plt.legend()

    # Save the plot
    plt.savefig("./data/growth_rate_predictions_tuned.png")
    plt.show()

if __name__ == "__main__":
    tune_and_build_models("./data/mock_poultry_data.csv", "./data/mock_poultry_data_with_predictions.csv")
