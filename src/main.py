from load_nc_data import load_pm25_nc
from preprocess import preprocess_data
from train_models import train_models
from plot_results import plot_predictions
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load data
    df = load_pm25_nc()

    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Train models
    lr, rf = train_models(X_train, y_train)

    # Predict
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Evaluate
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
    print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
    print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
    print("Random Forest R2:", r2_score(y_test, y_pred_rf))

    # Plot
    plot_predictions(y_test, y_pred_lr, y_pred_rf)

if __name__ == "__main__":
    main()
