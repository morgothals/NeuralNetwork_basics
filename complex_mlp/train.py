# complex_mlp.train.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics        import accuracy_score

from complex_mlp.model import ComplexMLP

def main():
    # 1) Adatok
    data = np.load("data/processed/dataset_preProcessed.npz")
    X = data["images"].reshape(len(data["images"]), -1)/255.0
    y = data["labels"]

    # 2) split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2,
        stratify=y, random_state=42
    )

    # 3) konfiguráció
    layer_sizes      = [768, 128, 64, 10]
    activations      = ["relu", "relu", "softmax"]
    loss_fn          = "cross_entropy"
    optimizer        = "sgd"
    #sgd: lr
    #momentum : lr , momentum
    #RMSProp : lr, beta ,eps
    #adam : lr, beta1, beta2, eps
    optimizer_params ={  "lr":0.001,
                         #"beta1":0.9,
                         #"beta2":0.999
                        }
    early_stop       = True
    patience         = 5
    tol              = 1e-4
    epochs           = 10
    batch_size       = 32

    # 4) modell
    model = ComplexMLP(
        layer_sizes=layer_sizes,
        activations=activations,
        loss=loss_fn,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_params,
        early_stopping=early_stop,
        patience=patience,
        tol=tol
    )

    # 5) tanítás
    history = model.train(
        X_train=X_tr, y_train=y_tr,
        X_val=X_val,   y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )

    # 6) kiértékelés
    y_pred = model.predict(X_val)
    print("Validation accuracy:", accuracy_score(y_val, y_pred))

    # 7) súlyok
    model.save("data/weights/complex_mlp_weights_2.npz")

if __name__ == "__main__":
    main()
