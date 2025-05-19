import os
from enum import Enum

import cv2 as cv
import numpy as np
import supervision as sv
from matplotlib import pyplot as plt


def load_imgs_from(folder_path: str) -> list[str]:
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    return images


class LoadMode(Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    UNCHANGED = "unchanged"


class Image:
    def __init__(self, data: np.ndarray, name: str):
        self.data = data
        self.name = name


def load_img(folder, name, mode=LoadMode.UNCHANGED):
    path = os.path.join(folder, name)

    modos = {
        LoadMode.RGB: cv.IMREAD_COLOR,
        LoadMode.BGR: cv.IMREAD_COLOR,
        LoadMode.GRAY: cv.IMREAD_GRAYSCALE,
        LoadMode.UNCHANGED: cv.IMREAD_UNCHANGED,
    }
    flag = modos.get(mode)

    image = cv.imread(path, flag)
    if mode == LoadMode.RGB:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    return image


def load_image(folder, name, mode=LoadMode.UNCHANGED):
    img = load_img(folder, name, mode)
    return Image(name = name, data = img)


def plot_images(images: [Image], grid_size=(3, 3)):
    images_grid = []
    titles = []

    for i, img in enumerate(images):
        images_grid.append(img.data)
        titles.append(img.name)

    sv.plot_images_grid(
        images=images_grid,
        titles=titles,
        grid_size=grid_size
    )


def plot_imgs_from(folder_path: str, grid_size=(3, 3), load_mode=LoadMode.UNCHANGED):
    images = load_imgs_from(folder_path)

    images_grid = []
    titles = []

    for i, name in enumerate(images):
        img = load_img(folder_path, name, load_mode)
        images_grid.append(img)
        titles.append(name)

    sv.plot_images_grid(
        images=images_grid,
        titles=titles,
        grid_size=grid_size
    )


def apply_white_patch(image_name, input_folder, output_folder, percentile=100):
    path = os.path.join(input_folder, image_name)
    image = cv.imread(path)

    # Separar canales BGR
    B, G, R = cv.split(image)

    # Convertir a float para operaciones precisas
    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

    # Calcular percentil por canal
    R_ref = np.percentile(R, percentile)
    G_ref = np.percentile(G, percentile)
    B_ref = np.percentile(B, percentile)

    # Evitar división por cero
    R_ref = R_ref if R_ref != 0 else 1
    G_ref = G_ref if G_ref != 0 else 1
    B_ref = B_ref if B_ref != 0 else 1

    # Normalización usando el percentil como referencia
    R_norm = np.round(255 * R / R_ref).clip(0, 255).astype(np.uint8)
    G_norm = np.round(255 * G / G_ref).clip(0, 255).astype(np.uint8)
    B_norm = np.round(255 * B / B_ref).clip(0, 255).astype(np.uint8)

    # Reconstruir imagen
    img_norm = cv.merge((B_norm, G_norm, R_norm))

    # Imprimir info
    print(f"{image_name}: R[{percentile}]={R_ref:.2f}, G[{percentile}]={G_ref:.2f}, B[{percentile}]={B_ref:.2f}")

    save_path = os.path.join(output_folder, image_name)
    cv.imwrite(save_path, img_norm)


def apply_white_patch_to(input_folder, output_folder, percentile=100):
    fixed_images = []
    for i, name in enumerate(load_imgs_from(input_folder)):
        fixed_images.append(apply_white_patch(name, input_folder, output_folder, percentile))


def plot_images_grid(images, titles, grid_size=(2, 3), figsize=(30, 15)):
    fig, axes = plt.subplots(*grid_size, figsize=figsize)
    axes = axes.flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    # Reducir espacios entre imágenes
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.show()


def plot_imgs_percentiles(input_folder, output_folder, percentiles=[100, 99, 97, 95, 90]):
    images = load_imgs_from(input_folder)
    for i, name in enumerate(images):
        images_grid = []
        titles = []

        img = load_img(input_folder, name)
        images_grid.append(img)
        titles.append(f'{name}[original]')

        for percentile in percentiles:
            img = load_img(f'{output_folder}/{percentile}', name)
            images_grid.append(img)
            titles.append(f'{name}[{percentile}]')

        plot_images_grid(images_grid, titles)


def plot_hist(image: Image, bins: int = 10) -> plt.Figure:
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

    if image.data.ndim == 2:
        # Escala de grises
        ax.hist(image.data.ravel(), bins=bins, color="gray", alpha=0.7)
        ax.set_title(f"{image.name} - Histograma (Grises) - {bins} bins")
    elif image.data.ndim == 3 and image.data.shape[2] == 3:
        # RGB
        colors = ("red", "green", "blue")
        for i, color in enumerate(colors):
            ax.hist(
                image.data[:, :, i].ravel(), bins=bins, color=color, alpha=0.5, label=color
            )
        ax.set_title(f"Histograma (RGB) - {bins} bins")
        ax.legend()
    else:
        raise ValueError("Imagen debe ser en escala de grises o RGB")

    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    ax.grid(True)
    plt.close(fig)
    return fig
