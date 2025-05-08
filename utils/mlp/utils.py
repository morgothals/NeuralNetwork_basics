import numpy as np
from complex_mlp.layers import Dense

def save_weights(layers, path: str):
    """
    Elmenti a Dense rétegek súlyait és bias-ait egy .npz fájlba.
    layers: a model.layers lista
    path: .npz fájl kiírási útvonala
    """
    params = {}
    idx = 0
    for layer in layers:
        if isinstance(layer, Dense):
            params[f"W{idx}"] = layer.W
            params[f"b{idx}"] = layer.b
            idx += 1
    np.savez_compressed(path, **params)
    print(f"Súlyok elmentve: {path}")

def load_weights(layers, path: str):
    """
    Betölti a korábban mentett súlyokat a .npz fájlból.
    layers: a model.layers lista
    path: .npz fájl beolvasási útvonala
    """
    data = np.load(path)
    idx = 0
    for layer in layers:
        if isinstance(layer, Dense):
            layer.W = data[f"W{idx}"]
            layer.b = data[f"b{idx}"]
            idx += 1
    print(f"Súlyok betöltve: {path}")
