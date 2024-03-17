# TODO: import dependencies and write unit tests below
from nn.nn import NeuralNetwork
import numpy as np
import pytest
import tensorflow as tf

@pytest.fixture
def setup_neural_network():
    # Basic NN architecture for testing; adjust as necessary
    nn_arch = [
        {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=2, epochs=100, loss_function='binary_cross_entropy')
    return nn

def test_single_forward(setup_neural_network):
    nn = setup_neural_network
    W_curr = np.array([[1, -1], [0, 1]])
    b_curr = np.array([[0], [1]])
    A_prev = np.array([[1, 2], [3, 4]])
    activation = 'relu'
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, activation)
    expected_A_curr = np.array([[0, 0], [4, 5]])
    expected_Z_curr = np.array([[-2, -2], [4, 5]])
    assert np.allclose(A_curr, expected_A_curr), "Single forward (A_curr) failed."
    assert np.allclose(Z_curr, expected_Z_curr), "Single forward (Z_curr) failed."

def test_forward(setup_neural_network):
    nn = setup_neural_network
    X = np.array([[1, 2], [3, 4]])
    output, cache = nn.forward(X)
    # Replace these expected values with those calculated based on your NN architecture and inputs
    expected_output_shape = (1, 2)
    assert output.shape == expected_output_shape, "Forward output shape mismatch."


def test_single_backprop(setup_neural_network):
    nn = setup_neural_network
    # Use mock values; these will need to be adjusted based on your network's architecture and expected outputs
    W_curr = np.array([[1, -1], [0, 1]])
    b_curr = np.array([[0]])
    Z_curr = np.array([[1, -1]])
    A_prev = np.array([[0.5, 0.5]])
    dA_curr = np.array([[1, -1]])
    activation_curr = 'relu'
    dA_prev, dW_curr, db_curr = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)
    # Replace these assertions with checks for expected gradients
    assert dA_prev.shape == A_prev.shape, "dA_prev shape mismatch."
    assert dW_curr.shape == W_curr.shape, "dW_curr shape mismatch."
    assert db_curr.shape == b_curr.shape, "db_curr shape mismatch."


def test_predict(setup_neural_network):
    nn = setup_neural_network
    X = np.array([[1, 2], [3, 4]])
    predictions = nn.predict(X)
    # Here you would check the shape of the predictions, and perhaps their values against expected outcomes
    expected_predictions_shape = (1, 2)  # Adjust based on your network's output layer
    assert predictions.shape == expected_predictions_shape, "Predictions shape mismatch."


def test_binary_cross_entropy(setup_neural_network):
    nn = setup_neural_network
    y_hat = np.array([[0.1, 0.9], [0.2, 0.8]])
    y = np.array([[0, 1], [0, 1]])
    loss = nn._binary_cross_entropy(y, y_hat)
    expected_loss = 0.164252033486018  # This is an example; calculate your expected loss based on your test case
    assert np.allclose(loss, expected_loss), "Binary cross entropy loss mismatch."


def test_binary_cross_entropy_backprop(setup_neural_network):
    nn = setup_neural_network
    y_hat = np.array([[0.1, 0.9], [0.8, 0.2]])
    y = np.array([[0, 1], [1, 0]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    # The expected derivative of binary cross entropy loss function for each prediction
    expected_dA = np.array([[1.11111111, -1.25], [-1.25, 1.11111111]])
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    print(bce_loss(y, y_hat))
    # Use np.allclose for floating point comparison as exact equality can be problematic due to precision issues
    assert np.allclose(dA, expected_dA, rtol=1e-6), "Binary cross entropy backpropagation calculation failed."

def test_mean_squared_error(setup_neural_network):
    nn = setup_neural_network
    y_hat = np.array([[0.1, 0.9], [0.2, 0.8]])
    y = np.array([[0, 1], [0, 1]])
    loss = nn._mean_squared_error(y, y_hat)
    # Calculate the expected loss based on your inputs
    expected_loss = 0.025
    assert np.allclose(loss, expected_loss), "Mean squared error calculation is incorrect."

def test_mean_squared_error_backprop(setup_neural_network):
    nn = setup_neural_network
    y_hat = np.array([[0.1, 0.9]])
    y = np.array([[0, 1]])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    # Calculate the expected dA based on your inputs
    expected_dA = np.array([[-0.9, -0.1]])
    assert np.allclose(dA, expected_dA), "Mean squared error backpropagation is incorrect."

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    pass