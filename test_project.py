import numpy as np
import pytest
from project import predict, loss, grad


def test_predict():
    """
    Tests the forward pass (Prediction).
    Math: Y = (w * X) + b
    """
    X = np.array([1.0, 2.0, 3.0])
    w = 2.0
    b = 1.0

    # Expected: (2 * 1)+1 = 3, (2 * 2)+1 = 5, (2 * 3)+1 = 7
    expected_output = np.array([3.0, 5.0, 7.0])

    # We use np.allclose instead of == to avoid floating-point rounding errors
    assert np.allclose(predict(X, w, b), expected_output)


def test_loss():
    # Tests the Mean Squared Error calculation.
    X = np.array([1.0, 2.0])
    Y = np.array([2.0, 4.0])

    assert np.isclose(loss(2.0, 0.0, X, Y), 0.0)
    assert np.isclose(loss(0.0, 0.0, X, Y), 5.0)


def test_grad():
    # Tests the backward pass (Calculus / Gradient Descent step).
    X = np.array([1.0, 2.0])
    Y = np.array([2.0, 4.0])
    w = 0.0
    b = 0.0
    learning_rate = 0.1

    new_w, new_b = grad(w, b, X, Y, learning_rate)

    assert np.isclose(new_w, 0.5)
    assert np.isclose(new_b, 0.3)
