import numpy as np
from .layers import Dense, ACTIVATIONS
from .losses import CrossEntropyLoss, MSELoss
from utils.mlp.utils import save_weights, load_weights

class ComplexMLP:
    def __init__(self,
                 layer_sizes,           # pl. [768,128,64,10]
                 activations,           # pl. ['relu','relu','softmax']
                 loss="cross_entropy",  # "cross_entropy" vagy "mse"
                 early_stopping=False,
                 patience=5,
                 tol=1e-4):
        assert len(layer_sizes)-1 == len(activations)
        self.layers = []
        for i in range(len(activations)):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(ACTIVATIONS[activations[i]]())
        # loss
        if loss=="mse":
            self.loss_fn = MSELoss()
        else:
            self.loss_fn = CrossEntropyLoss()
        # early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def train(self, X_train, y_train,
                    X_val=None, y_val=None,
                    epochs=50,
                    batch_size=32,
                    lr=0.01):
        history = {"train_loss":[], "val_loss":[]}
        best_val = np.inf
        wait = 0
        best_weights = None

        for ep in range(1, epochs+1):
            # shuffle
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

            # epoch train
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]

                preds = self.forward(xb)
                loss = self.loss_fn.forward(preds, yb)
                grad = self.loss_fn.backward(lr)
                self.backward(grad, lr)

                total_loss += loss
            avg_train = total_loss / (len(X_train)/batch_size)
            history["train_loss"].append(avg_train)

            # validation
            if X_val is not None:
                vp = self.forward(X_val)
                vloss = self.loss_fn.forward(vp, y_val)
                history["val_loss"].append(vloss)

                # early stop
                if self.early_stopping:
                    if best_val - vloss > self.tol:
                        best_val = vloss
                        wait = 0
                        # save best
                        best_weights = [ (l.W.copy(), l.b.copy()) 
                                         for l in self.layers if hasattr(l,"W") ]
                    else:
                        wait += 1
                        if wait >= self.patience:
                            print(f"Early stopping at epoch {ep}")
                            # restore
                            wi = 0
                            for l in self.layers:
                                if hasattr(l,"W"):
                                    l.W, l.b = best_weights[wi]
                                    wi+=1
                            return history
                print(f"Epoch {ep}/{epochs}  train_loss={avg_train:.4f}  val_loss={vloss:.4f}")
            else:
                print(f"Epoch {ep}/{epochs}  train_loss={avg_train:.4f}")

        return history

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=1)
    
    def save(self, path: str):
        """Publikus metódus: elmenti a Dense súlyokat .npz-be."""
        save_weights(self.layers, path)

    def load(self, path: str):
        """Publikus metódus: betölti a Dense súlyokat .npz-ből."""
        load_weights(self.layers, path)
