# complex_mlp.model.py
import numpy as np
from complex_mlp.layers    import Dense, ACTIVATIONS
from complex_mlp.losses    import CrossEntropyLoss, MSELoss
from complex_mlp.optimizer import SGD, Momentum, RMSProp, Adam
from utils.mlp.utils       import save_weights, load_weights

class ComplexMLP:
    def __init__(self,
                 layer_sizes,            # pl. [768,128,64,10]
                 activations,            # pl. ['relu','relu','softmax']
                 loss="cross_entropy",   # "cross_entropy" vagy "mse"
                 optimizer="sgd",        # "sgd","momentum","rmsprop","adam"
                 optimizer_kwargs=None,  # pl. {'lr':0.005,'momentum':0.9}
                 early_stopping=False,
                 patience=5,
                 tol=1e-4):
        assert len(layer_sizes)-1 == len(activations), \
               "layer_sizes-1 == activations hossza"

        # --- Rétegek felépítése ---
        self.layers = []
        for i, act in enumerate(activations):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(ACTIVATIONS[act]())

        # --- Loss kiválasztása ---
        self.loss_fn = MSELoss() if loss=="mse" else CrossEntropyLoss()

        # --- Optimizer példányosítása ---
        opts = optimizer_kwargs or {}
        if optimizer.lower() == "sgd":
            self.optimizer = SGD(**opts)
        elif optimizer.lower() == "momentum":
            self.optimizer = Momentum(**opts)
        elif optimizer.lower() == "rmsprop":
            self.optimizer = RMSProp(**opts)
        elif optimizer.lower() == "adam":
            self.optimizer = Adam(**opts)
        else:
            raise ValueError(f"Ismeretlen optimizer: {optimizer}")

        # --- Early stopping beállítások ---
        self.early_stopping = early_stopping
        self.patience       = patience
        self.tol            = tol

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        # végigmegyünk a rétegeken, kigyűjtjük dW, db
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        # optimizer frissítés
        self.optimizer.step(self.layers)
        return grad

    def train(self,
              X_train, y_train,
              X_val=None, y_val=None,
              epochs=50,
              batch_size=32):
        history  = {"train_loss":[], "val_loss":[]}
        best_val = np.inf
        wait     = 0
        best_ws  = None

        for ep in range(1, epochs+1):
            # shuffle
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

            # --- train batch-ek ---
            train_loss = 0
            for i in range(0, len(X_train), batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]

                preds = self.forward(xb)
                loss  = self.loss_fn.forward(preds, yb)
                grad  = self.loss_fn.backward()
                self.backward(grad)

                train_loss += loss

            avg_tr = train_loss / (len(X_train)/batch_size)
            history["train_loss"].append(avg_tr)

            # --- validation ---
            if X_val is not None:
                vp    = self.forward(X_val)
                vloss = self.loss_fn.forward(vp, y_val)
                history["val_loss"].append(vloss)
                print(f"Epoch {ep}/{epochs}  train_loss={avg_tr:.4f}  val_loss={vloss:.4f}")

                if self.early_stopping:
                    if best_val - vloss > self.tol:
                        best_val = vloss
                        wait     = 0
                        # legjobb súlyok tárolása
                        best_ws = [(l.W.copy(), l.b.copy())
                                   for l in self.layers if hasattr(l, "W")]
                    else:
                        wait += 1
                        if wait >= self.patience:
                            print(f"Early stopping at epoch {ep}")
                            # visszaállítjuk a best_ws-t
                            wi = 0
                            for l in self.layers:
                                if hasattr(l, "W"):
                                    l.W, l.b = best_ws[wi]
                                    wi += 1
                            return history
            else:
                print(f"Epoch {ep}/{epochs}  train_loss={avg_tr:.4f}")

        return history

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=1)

    def save(self, path):
        save_weights(self.layers, path)

    def load(self, path):
        load_weights(self.layers, path)
