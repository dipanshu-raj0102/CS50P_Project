# COMMAND-LINE LINEAR REGRESSION MODEL

#### Video Demo: <INSERT_YOUTUBE_URL_HERE>

#### Description:

This project is a Command-Line Machine Learning tool built from scratch to satisfy the requirements of the CS50P Final Project. Instead of relying on high-level machine learning libraries like `scikit-learn` to abstract the logic, this project implements a Linear Regression engine using pure mathematical principles.

## Architecture and Files

The project is structured strictly to adhere to the CS50P rubric:

- `project.py`: The core application engine that houses the `main()` function. It contains the following custom functions:
  - `load_dataset(filepath)`: Reads the dataset, extracts features and targets, and applies Feature Scaling.
  - `predict(X, w, b)`: Calculates the predicted values given the feature matrix, weight, and bias.
  - `loss(w, b, X, Y)`: Calculates the Mean Squared Error (MSE) between the predictions and the actual truth.
  - `grad(w, b, X, Y, learning_rate)`: Calculates the partial derivatives and updates the weight and bias.
  - `train_model(learning_rate, iteration)`: The training loop that orchestrates the data loading, predicting, measuring, and learning.
- `test_project.py`: A comprehensive test suite built for `pytest`. Because the mathematical operations (`predict`, `loss`, `grad`) were extracted into pure functions, this file rigorously tests the logic in isolation without relying on opening the CSV file.
- `requirements.txt`: Defines the external environment dependencies, specifically `numpy` and `pandas`.

## Design Decisions and Engineering Challenges

**1. Overcoming the Exploding Gradient:**
During initial development, the model suffered from the "Exploding Gradient" problem, where the loss would skyrocket to infinity (`inf`) and `NaN` because the raw input features contained large numbers. Instead of brute-forcing a microscopic learning rate, **Feature Scaling (Standardization/Z-Score)** was implemented. By compressing the data using the mean and standard deviation, the gradient stabilized, allowing the model to smoothly converge.

**2. Vectorization over Procedural Loops:**
While standard Python `for` loops can calculate predictions, this project utilizes `numpy` for Vectorization. This allows the computer to perform mathematical operations on entire arrays at once, making the model computationally efficient and scalable.

**3. Separation of Concerns (DRY Principle):**
The mathematical formula for the forward pass was abstracted into its own `predict()` function. This prevents repeating the same equation inside both the `loss` and `grad` functions, honoring the DRY (Don't Repeat Yourself) principle and creating a single source of truth.

## Usage

1. Install requirements: `pip install -r requirements.txt`
2. Execute tests: `pytest test_project.py`
3. Run the model: `python project.py`
