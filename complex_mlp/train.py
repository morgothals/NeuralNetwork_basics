import numpy as np
from sklearn.model_selection import train_test_split
from .model import ComplexMLP

# 1) adatbetöltés
data = np.load("data/processed/dataset_preProcessed.npz")
X = data["images"].reshape(len(data["images"]), -1)/255.
y = data["labels"]

# 2) train/val split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Tanító: {len(X_tr)}, Validálás: {len(X_val)}")

# 3) konfiguráció
layer_sizes   = [768, 128, 64, 10]
activations   = ["relu","relu","softmax"]
loss_fn       = "cross_entropy"   # vagy "mse"
early_stop    = True
patience      = 5
tol           = 1e-4
epochs        = 50
batch_size    = 32
learning_rate = 0.005

# 4) modell
model = ComplexMLP(layer_sizes, activations,
                  loss=loss_fn,
                  early_stopping=early_stop,
                  patience=patience,
                  tol=tol)

# 5) tanítás
history = model.train(X_tr, y_tr,
                      X_val, y_val,
                      epochs=epochs,
                      batch_size=batch_size,
                      lr=learning_rate)

# 6) kiértékelés
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, y_pred))

model.save('data/weights/complex_mlp_weights.npz')
