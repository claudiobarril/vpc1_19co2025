import cv2
import os
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class LoadMode(Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    UNCHANGED = "unchanged"


def load_image_cv2(image_path: str, mode: LoadMode = LoadMode.GRAY) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada en: {image_path}")

    modos = {
        LoadMode.RGB: cv2.IMREAD_COLOR,
        LoadMode.BGR: cv2.IMREAD_COLOR,
        LoadMode.GRAY: cv2.IMREAD_GRAYSCALE,
        LoadMode.UNCHANGED: cv2.IMREAD_UNCHANGED,
    }
    flag = modos.get(mode)
    image = cv2.imread(image_path, flag)
    if mode == LoadMode.RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def read_image_cv2(image: np.ndarray, window_title: str = "Imagen") -> None:
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_hist(image: np.ndarray, bins: int = 10) -> plt.Figure:
    """
    Genera un histograma de intensidad para una imagen en escala de grises o RGB.

    Parámetros
    ----------
    image : np.ndarray
        Imagen en escala de grises (2D) o RGB (3D con 3 canales).
    bins : int
        Número de bins del histograma.

    Retorna
    -------
    plt.Figure
        Figura de matplotlib con el histograma.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if image.ndim == 2:
        # Escala de grises
        ax.hist(image.ravel(), bins=bins, color="gray", alpha=0.7)
        ax.set_title(f"Histograma (Grises) - {bins} bins")
    elif image.ndim == 3 and image.shape[2] == 3:
        # RGB
        colors = ("red", "green", "blue")
        for i, color in enumerate(colors):
            ax.hist(
                image[:, :, i].ravel(), bins=bins, color=color, alpha=0.5, label=color
            )
        ax.set_title(f"Histograma (RGB) - {bins} bins")
        ax.legend()
    else:
        raise ValueError("Imagen debe ser en escala de grises o RGB")

    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    ax.grid(True)
    return fig


def make_multiple_hist(
    image: np.ndarray, bins_list: list[int], output_dir: str
) -> None:
    """
    Genera y guarda múltiples histogramas de una imagen, variando el número de bins.

    Parámetros
    ----------
    image : np.ndarray
        Imagen de entrada (grises o RGB).
    bins_list : list[int]
        Lista con la cantidad de bins para cada histograma.
    output_dir : str
        Carpeta donde se guardarán las imágenes de los histogramas.
    """
    os.makedirs(output_dir, exist_ok=True)

    for bins in bins_list:
        fig = make_hist(image, bins=bins)
        save_path = os.path.join(output_dir, f"hist_{bins}_bins.png")
        fig.savefig(save_path)
        plt.close(fig)
