#TODO Összehasonlítani a két dataset et (sima, preprocessed)

# simplemlp/train.py
import numpy as np
#from simplemlp.model import SimpleMLP
from model import SimpleMLP
from sklearn.model_selection import train_test_split

# Dataset betöltése
data = np.load("data/processed/dataset_preProcessed.npz")
images = data["images"]
labels = data["labels"]

# Lapítás és normalizálás [0,1] közé
X = images.reshape(len(images), -1) / 255.0
y = labels

# Train/test split (80% tanítás, 20% teszt)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tanító minták: {len(X_train)}, Teszt minták: {len(X_test)}")

# Modell inicializálása
model = SimpleMLP()

#Elmentett súlyok betöltése
model.load_weights('data/weights/simplemlp_weights.npz')

# Tanítási paraméterek
epochs = 5
batch_size = 32
learning_rate = 0.01

print("Tanítás indul...")
for epoch in range(epochs):
    # Véletlen sorrend minden epoch elején
    indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]

    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        loss = model.train_step(x_batch, y_batch, learning_rate)
        total_loss += loss

    avg_loss = total_loss / (len(X_train) / batch_size)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Kiértékelés tesztadaton
print("\nTesztelés...")
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Pontosság a teszthalmazon: {accuracy:.2%}")

#Súlyok mentése
model.save_weights('data/weights/simplemlp_weights.npz')