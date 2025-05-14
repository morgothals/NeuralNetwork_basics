# complex_mlp.layers.py
import numpy as np

class Layer:
    """Absztrakt alaposztály minden rétegnek."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Visszaadja dL/dinput értéket, eltárolja a saját paraméter-gradienseket."""
        raise NotImplementedError
    def params_and_grads(self):
        """
        Generator, ami visszaadja a (paraméter, paraméter_grad) párokat.
        Ezt használja majd az optimizer.
        """
        return
        yield

class Dense(Layer):
    def __init__(self, in_features: int, out_features: int):
        # W shape=(in, out), b shape=(1, out)
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
        # hogy tartsuk a grad-eket minden backward után
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x shape = (N, in_features)
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: dL/dout, shape = (N, out_features)
        # kiszámoljuk a gradienseket
        self.dW = self.x.T @ grad       # shape = (in, out)
        self.db = np.sum(grad, axis=0, keepdims=True)  # shape = (1, out)
        # visszaadjuk dL/dx-et
        return grad @ self.W.T

    def params_and_grads(self):
        # itt adja vissza a paramétereket és grad-jeiket
        yield self.W, self.dW
        yield self.b, self.db

class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0).astype(float)
        return x * self.mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask

class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.y*(1-self.y))

class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1 - self.y**2)

class Softmax(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        self.y = e / e.sum(axis=1, keepdims=True)
        return self.y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Softmax + CrossEntropy kombóra elegendő, hogy továbbadjuk a grad-et
        return grad

# aktivációk egyszerű létrehozásához
ACTIVATIONS = {
    "relu":   ReLU,
    "sigmoid":Sigmoid,
    "tanh":   Tanh,
    "softmax":Softmax
}
