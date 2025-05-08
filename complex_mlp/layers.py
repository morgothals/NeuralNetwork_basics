import numpy as np

class Dense:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, grad, lr):
        # grad: dL/dy, x: (N, in), W: (in, out)
        dW = self.x.T @ grad
        db = np.sum(grad, axis=0, keepdims=True)
        dx = grad @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class ReLU:
    def forward(self, x):
        self.mask = (x > 0).astype(float)
        return x * self.mask
    def backward(self, grad, lr):
        return grad * self.mask

class Sigmoid:
    def forward(self, x):
        self.y = 1/(1+np.exp(-x))
        return self.y
    def backward(self, grad, lr):
        return grad * (self.y*(1-self.y))

class Tanh:
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y
    def backward(self, grad, lr):
        return grad * (1 - self.y**2)

class Softmax:
    def forward(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        self.y = e / e.sum(axis=1, keepdims=True)
        return self.y
    def backward(self, grad, lr):
        # Identity here; full grad from loss
        return grad

# mapping
ACTIVATIONS = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax
}
