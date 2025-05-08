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


# Szín-címke megfeleltetés (csak ahol leírás volt)
CLASS_COLOR_MAP = {
    (255, 255, 255): 0,  # Jól futható erdő
    (62, 255, 23): 1,    # Leküzdhető növényzet
    (197, 255, 186): 2,  # Lassan futható erdő
    (139, 255, 116): 3,  # Nehezen futható erdő
    (255, 204, 104): 4,  # Nyílt terület (alap)
    (255, 187, 54): 5,   # Nyílt terület (erősebb)
    (255, 221, 155): 6,  # Durva nyílt terület
    (128, 255, 255): 7,  # Sekély vízfelület
    (0, 255, 255): 8,    # Áthatolhatatlan vízfelület
   # (0, 0, 0): 9         # Fekete – utak, sziklák
}

def classify_tile_exact(tile_rgb):
    """
    Meghatározza a tile leggyakoribb színét, és az előre definiált címkéhez rendeli.
    """
    rgb_data = np.array(tile_rgb).reshape(-1, 3)
    rgb_tuples = [tuple(pixel) for pixel in rgb_data]
    most_common_rgb, _ = Counter(rgb_tuples).most_common(1)[0]

    return CLASS_COLOR_MAP.get(most_common_rgb, -1)



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

            label = classify_tile_exact(tile.convert('RGB'))
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
    image_path='data/raw/tajfuto_normalized.png',
    output_path='data/processed/dataset_preProcessed.npz'
)
