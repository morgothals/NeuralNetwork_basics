# Instabilitást okozott ezért cserélve lett a másikra
# simplemlp/layers.py
import numpy as np

# Dense réteg
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        grad_weights = self.input.T.dot(grad_output)
        grad_bias = grad_output.mean(axis=0)*self.input.shape[0]
        
        # Súlyok frissítése
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_output.dot(self.weights.T)

# ReLU aktiváció
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output, learning_rate):
        if np.any(np.isnan(self.input)):
            raise ValueError("NaN található a ReLU inputban")
        relu_grad = self.input > 0
        return grad_output * relu_grad

# Softmax réteg (kimenetre)
class Softmax:
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output, learning_rate):
        return grad_output  # A gradienst a cross-entropy lossból kapjuk meg

# Cross-entropy veszteségfüggvény
def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    correct_confidences = predictions[range(N), targets]
    loss = -np.log(correct_confidences + 1e-9).mean()
    return loss

def cross_entropy_grad(predictions, targets):
    grad = predictions.copy()
    grad[range(len(targets)), targets] -= 1
    grad /= len(targets)
    return grad
