from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_models(X_train, y_train):
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Save models
    os.makedirs("../results/models", exist_ok=True)
    with open("../results/models/linear_regression.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open("../results/models/random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)

    return lr, rf
