import os
import numpy as np
from PIL import Image
from collections import Counter

# RGB színhatárok különböző osztályokhoz
def classify_tile(tile_rgb):
    rgb_data = np.array(tile_rgb).reshape(-1, 3)  # (256, 3)
    rgb_tuples = [tuple(pixel) for pixel in rgb_data]
    most_common_rgb, _ = Counter(rgb_tuples).most_common(1)[0]
    r, g, b = most_common_rgb

    # Fehér: jól futható erdő
    if r > 235 and g > 235 and b > 235:
        return 0

    # Zöld árnyalatok: futhatóság szerint
   
    if g > 230 and 210 > r > 180 and 200 > b > 160:
        return 1  # világos zöld – jól futható erdőaljnövényzet
    elif g > 210 and 160 > r >= 105 and 135 > b >= 80:
        return 2  # közép zöld – lassabb futhatóság
    elif g > 205 and 40 < r < 80 and b < 40:
        return 3  # sötét zöld – bozót, nagyon rossz futhatóság
    
     # Halvány Sárga: durva nyílt terület
    elif r> 240 and g > 200 and 140 < b < 180: 
        return 9

    # Sárga: nyílt terület
    elif r > 240 and 220 >g > 170 and 60 < b < 100:
        return 4

    # Kék: vízfelület
    elif b > 150 and b > r and b > g:
        return 5

    # Fekete: utak, szikla
    elif r < 20 and g < 20 and b < 20:
        return 6

    # Bíbor/lila: tiltott terület, pályajel
    elif r > 150 and b > 150 and g < 100:
        return 7

    # Barna: domborzat
    elif 230 > r > 170 and  70 <g < 120 and b < 50:
        return 8

    return -1  # nem egyértelmű


def generate_dataset(image_path, output_path, tile_size=16):
    image = Image.open(image_path).convert('RGBA')
    w, h = image.size

    tiles, labels = [], []

    for i in range(0, w, tile_size):
        for j in range(0, h, tile_size):
            tile = image.crop((i, j, i + tile_size, j + tile_size))
            if tile.size != (tile_size, tile_size):
                continue
            alpha = np.array(tile)[..., 3]
            if np.all(alpha == 0):
                continue  # kihagyjuk az üres területeket

            label = classify_tile(tile.convert('RGB'))
            if label != -1:
                tiles.append(np.array(tile.convert('RGB')))
                labels.append(label)

    tiles = np.array(tiles)
    labels = np.array(labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, images=tiles, labels=labels)
    print(f"{len(tiles)} csempe mentve ide: {output_path}")

# Használat:
generate_dataset(
    image_path='data/raw/terkep_vagasra_1.png',
    output_path='data/processed/dataset.npz'
)
