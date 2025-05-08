from PIL import Image
import numpy as np
from color_palette import PREDEFINED_RGB_COLORS
from skimage.color import rgb2lab, lab2rgb

def normalize_image_colors(image: Image.Image) -> Image.Image:
    """ Lecseréli az összes pixelt a legközelebbi előre definiált színre LAB színtér alapján. """
    img = image.convert("RGB")
    data = np.array(img)  # shape (H, W, 3)
    h, w, _ = data.shape

    # Átalakítjuk LAB színtérbe (jobb vizuális hasonlóság szerint)
    pixel_lab = rgb2lab(data / 255.0).reshape(-1, 3)
    palette_lab = rgb2lab(PREDEFINED_RGB_COLORS[None, ...] / 255.0)[0]

    # Minden pixelhez megkeressük a legközelebbi színt
    idxs = np.argmin(np.linalg.norm(pixel_lab[:, None, :] - palette_lab[None, :, :], axis=2), axis=1)
    matched_rgb = PREDEFINED_RGB_COLORS[idxs]

    new_data = matched_rgb.reshape(h, w, 3).astype(np.uint8)
    return Image.fromarray(new_data)





image = Image.open("data/raw/terkep_vagasra_1.png")
normalized = normalize_image_colors(image)
normalized.save("data/processed/tajfuto_normalized.png")

