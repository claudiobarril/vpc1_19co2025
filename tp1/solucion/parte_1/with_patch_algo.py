from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import os
from util import load_image


def white_patch_algo(image, percentile: int = 100) -> np.ndarray:
    """
    Devuelve una imagen corregida mediante balance de blancos
    utilizando el algoritmo White Patch.

    Parámetros
    ----------
    image : numpy array
        Imagen a procesar con el algoritmo White Patch.
    percentile : int, opcional
        Valor percentil a considerar como máximo por canal.
    """

    white_patch_image = img_as_ubyte(
        (image * 1.0 / np.percentile(image, percentile, axis=(0, 1))).clip(0, 1)
    )
    return white_patch_image


def plot_white_patch_comparison(
    original_image: np.ndarray,
    corrected_image: np.ndarray,
    save_path: str | None = None,
) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(original_image)
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    ax[1].imshow(corrected_image, cmap="gray")
    ax[1].set_title("Imagen Corregida (White Patch)")
    ax[1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


image = load_image("material/white_patch/test_blue.png")
corrected = white_patch_algo(image)
plot_white_patch_comparison(
    image, corrected, save_path="material/white_patch/solutions/comparacion.png"
)
