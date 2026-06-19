import numpy as np
import pandas as pd

path = "./tvmarketing.csv"


def load_dataset(filepath):
    # Reads the CSV, extracts features and targets, and applies Feature Scaling.

    data = pd.read_csv(filepath)
    X = data["TV"]
    Y = data["Sales"]
    return X, Y


def predict(X, w, b):
    # Calculates the predicted values (Y_pred) given the feature matrix X, weight w, and bias b.
    return (w * X) + b


def loss(w, b, X, Y):
    # Calculates the Mean Squared Error (MSE) between the predictions and the actual truth.

    return 1 / (2 * len(Y)) * np.sum((w * X + b - Y) ** 2)


def grad(w, b, X, Y, learning_rate):
    # Calculates the partial derivatives and updates the weight and bias.

    dloss_db = 1 / len(X) * np.sum(w * X + b - Y)
    dloss_dw = 1 / len(X) * np.dot(w * X + b - Y, X)

    w_n = w - learning_rate * dloss_dw
    b_n = b - learning_rate * dloss_db

    w = w_n
    b = b_n

    return w, b


def train_model(learning_rate=0.01, iteration=100):
    # The training loop that orchestrates the data loading, predicting, measuring, and learning.

    w = 0.0
    b = 0.0

    X, Y = load_dataset(path)
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)

    for i in range(iteration):
        los = loss(w, b, X, Y)
        print(f"Loss in iteration {i} is {los}")
        w, b = grad(w, b, X, Y, learning_rate)

    return w, b


def main():
    print("--- Training Linear Regression Model ---")

    w_gd, b_gd = train_model(0.1, 100)

    print("\n--- Training Complete ---")

    X_pred = np.array([50, 120, 280])

    data = pd.read_csv(path)
    X = data["TV"]
    Y = data["Sales"]

    X_pred_norm = (X_pred - np.mean(X)) / np.std(X)
    Y_pred_gd_norm = w_gd * X_pred_norm + b_gd

    Y_pred_gd = Y_pred_gd_norm * np.std(Y) + np.mean(Y)

    print(f"TV marketing expenses:\n{X_pred}")
    print(f"Predictions of sales according to trained model:\n{Y_pred_gd}")


if __name__ == "__main__":
    main()
