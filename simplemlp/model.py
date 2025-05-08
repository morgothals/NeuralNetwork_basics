# simplemlp/model.py
#from simplemlp.layers import Dense, ReLU, Softmax, cross_entropy, cross_entropy_grad
from layers import Dense, ReLU, Softmax, cross_entropy, cross_entropy_grad
import numpy as np

class SimpleMLP:
    def __init__(self):
        self.layers = [
            Dense(768, 128),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, 10),
            Softmax()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

    def train_step(self, x_batch, y_batch, learning_rate):
        predictions = self.forward(x_batch)
        loss = cross_entropy(predictions, y_batch)
        grad_output = cross_entropy_grad(predictions, y_batch)
        self.backward(grad_output, learning_rate)
        return loss

    def predict(self, x):
        predictions = self.forward(x)
        return np.argmax(predictions, axis=1)
    
    def save_weights(self, path: str):
        """Elmenti a Dense rétegek súlyait és bias-ait egy .npz fájlba."""
        params = {}
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                params[f"W{idx}"] = layer.weights
                params[f"b{idx}"] = layer.bias
                idx += 1
        np.savez_compressed(path, **params)
        print(f"Súlyok elmentve: {path}")

    def load_weights(self, path: str):
        """Betölti a korábban mentett súlyokat a .npz fájlból."""
        data = np.load(path)
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = data[f"W{idx}"]
                layer.bias    = data[f"b{idx}"]
                idx += 1
        print(f"Súlyok betöltve: {path}")
