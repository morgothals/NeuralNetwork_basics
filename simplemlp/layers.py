# simplemlp/layers.py
import numpy as np

# Dense réteg Xavier-inicializációval és gradient átlagolással
class Dense:
    def __init__(self, input_size, output_size):
        # Glorot/Xavier inicializáció
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias    = np.zeros(output_size)

    def forward(self, x):
        self.input = x               # (batch_size, input_size)
        return np.dot(x, self.weights) + self.bias  # (batch_size, output_size)

    def backward(self, grad_output, learning_rate):
        # grad_output: (batch_size, output_size)
        batch_size = self.input.shape[0]

        # Átlagolt gradiensek
        grad_weights = self.input.T.dot(grad_output) / batch_size   # (input_size, output_size)
        grad_bias    = grad_output.mean(axis=0)                     # (output_size,)

        # Súlyok frissítése
        self.weights -= learning_rate * grad_weights
        self.bias    -= learning_rate * grad_bias

        # visszapropagáljuk a gradienst a bemenet felé
        return grad_output.dot(self.weights.T)  # (batch_size, input_size)

# ReLU aktiváció
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output, learning_rate):
        relu_grad = (self.input > 0).astype(float)
        return grad_output * relu_grad

# Softmax réteg (kimenetre)
class Softmax:
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output, learning_rate):
        # A cross-entropy grad már tartalmazza a softmax deriváltját
        return grad_output

# Cross-entropy veszteségfüggvény
def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    correct_confidences = predictions[np.arange(N), targets]
    log_likelihoods = -np.log(correct_confidences + 1e-9)
    return log_likelihoods.mean()

def cross_entropy_grad(predictions, targets):
    N = predictions.shape[0]
    grad = predictions.copy()
    grad[np.arange(N), targets] -= 1
    return grad / N
