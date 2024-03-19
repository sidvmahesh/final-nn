# TODO: import dependencies and write unit tests below
from nn.nn import NeuralNetwork
from nn.preprocess import one_hot_encode_seqs, sample_seqs
import numpy as np
import pytest
import tensorflow as tf
import random

@pytest.fixture
def setup_neural_network():
    nn_arch = [
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=2, epochs=100, loss_function='binary_cross_entropy')
    return nn

def test_single_forward(setup_neural_network):
    nn = setup_neural_network
    W_curr = np.array([[1, -1], [0, 1]])
    b_curr = np.array([[0], [1]])
    A_prev = np.array([[1, 2], [3, 4]])
    activation = 'sigmoid'
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, activation)
    expected_A_curr = np.array([[0.119, 0.119], [0.982, 0.993]])
    expected_Z_curr = np.array([[-2, -2], [4, 5]])
    print(A_curr, Z_curr)
    assert np.allclose(A_curr, expected_A_curr, rtol = 1e-2), "Single forward (A_curr) failed."
    assert np.allclose(Z_curr, expected_Z_curr, rtol = 1e-2), "Single forward (Z_curr) failed."

def test_forward(setup_neural_network):
    nn = setup_neural_network
    X = np.array([[1, 2], [3, 4]])
    output, cache = nn.forward(X)
    expected_output_shape = (1, 2)
    assert output.shape == expected_output_shape, "Forward output shape mismatch."


def test_single_backprop(setup_neural_network):
    nn = setup_neural_network
    W_curr = np.array([[1, -1]])
    b_curr = np.array([[0]])
    Z_curr = np.array([[1, -1]])
    A_prev = np.array([[0.5, 0.5]])
    dA_curr = np.array([[1, -1]])
    activation_curr = 'sigmoid'
    dA_prev, dW_curr, db_curr = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)
    assert dA_curr.shape == (1, 2)
    assert dW_curr.shape == (1, 1)
    assert db_curr.shape == (1, 1)
    assert True

def test_predict(setup_neural_network):
    nn = setup_neural_network
    X = np.array([[1, 2], [3, 4]])
    predictions = nn.predict(X)
    expected_predictions_shape = (1, 2)
    assert predictions.shape == expected_predictions_shape, "Predictions shape mismatch."


def test_binary_cross_entropy(setup_neural_network):
    nn = setup_neural_network
    y_hat = np.array([[0.1, 0.9], [0.2, 0.8]])
    y = np.array([[0, 1], [0, 1]])
    loss = nn._binary_cross_entropy(y, y_hat)
    expected_loss = 0.164252033486018
    assert np.allclose(loss, expected_loss), "Binary cross entropy loss mismatch."


def test_binary_cross_entropy_backprop(setup_neural_network):
    nn = setup_neural_network
    y_hat = np.array([[0.1, 0.9], [0.8, 0.2]])
    y = np.array([[0, 1], [1, 0]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    expected_dA = np.array([[0.5555, -0.5555], [-0.625, 0.625]])
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    print(bce_loss(y, y_hat))
    print(dA)
    assert np.allclose(dA, expected_dA, rtol=1e-2), "Binary cross entropy backpropagation calculation failed."

def test_mean_squared_error(setup_neural_network):
    nn = setup_neural_network
    y_hat = np.array([[0.1, 0.9], [0.2, 0.8]])
    y = np.array([[0, 1], [0, 1]])
    loss = nn._mean_squared_error(y, y_hat)
    expected_loss = [0.0125, 0.0125]
    print(loss)
    assert np.allclose(loss, expected_loss), "Mean squared error calculation is incorrect."

def test_mean_squared_error_backprop(setup_neural_network):
    nn = setup_neural_network
    y_hat = np.array([[0.1, 0.9]])
    y = np.array([[0, 1]])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    print(dA)
    expected_dA = np.array([[0.2, -0.1]])
    assert np.allclose(dA, expected_dA), "Mean squared error backpropagation is incorrect."

def test_sample_seqs():
    nucleotides = "ATCG"
    positives = [''.join(random.choices(nucleotides, k=17)) for _ in range(10)]
    positive_labels = [True for i in positives]
    negatives = [''.join(random.choices(nucleotides, k=100)) for _ in range(50)]
    negative_labels = [False for i in negatives]
    sequences = positives
    labels = positive_labels
    sequences.extend(negatives)
    labels.extend(negative_labels)
    result_seqs, result_labels = sample_seqs(seqs = sequences, labels = labels)
    assert sum(result_labels) == len(result_labels) / 2
    assert min([len(i) for i in result_seqs]) == max([len(i) for i in result_seqs])

def test_one_hot_encode_seqs():
    seq_arr = []
    assert one_hot_encode_seqs(seq_arr) == []
    seq_arr = ['ATCG', 'GCA']
    expected_output = [[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], [0,0,0,1, 0,0,1,0, 1,0,0,0]]
    assert one_hot_encode_seqs(seq_arr) == expected_output