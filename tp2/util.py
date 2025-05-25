import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from typing import Tuple


def show_image(image: np.ndarray, title: str = ""):
    """
    Muestra una imagen en una celda usando matplotlib.

    Parámetros:
    - imagen: Imagen a mostrar, puede estar en escala de grises (2D) o en color (3D en BGR).
    - title: Título a mostrar sobre la imagen.

    Nota:
    Si la imagen está en formato BGR (como es usual con OpenCV), se convierte a RGB para mostrar correctamente los colores.
    """
    plt.figure(figsize=(5, 4))
    if len(image.shape) == 3:
        imagen_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(imagen_rgb)
    else:
        plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_frame(
    video_path: str,
    frame_target: int,
    *,
    to_gray: bool = False,
) -> np.ndarray:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video_path!r}")

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if frame_target < 0 or frame_target >= total:
        cap.release()
        raise ValueError(f"Frame index {frame_target} Fuera de rango (0-{total-1})")

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_target)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"No se pudo leer frame:  {frame_target}")

    # Mostrar ambas versiones
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(frame_rgb)
    axs[0].set_title(f"Frame {frame_target} – Color")
    axs[0].axis("off")

    axs[1].imshow(frame_gray, cmap="gray")
    axs[1].set_title(f"Frame {frame_target} – Gris")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

    return frame_gray if to_gray else frame


def show_n_frames(video_path: str, n: int):
    """
    Lee y muestra los primeros 'n' frames de un video, en versión original (color)
    y escala de grises, una al lado de la otra.

    Parámetros:
    - video_path: Ruta al archivo de video.
    - n: Número de frames a mostrar.

    Cada par de imágenes se muestra en una sola figura: izquierda (color), derecha (gris).
    """
    captura_video = cv.VideoCapture(video_path)
    if not captura_video.isOpened():
        print("Error al abrir el archivo de video")
        return

    for i in range(n):
        ret, frame = captura_video.read()
        if not ret:
            break

        frame_gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Mostrar lado a lado
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        axes[0].imshow(frame_rgb)
        axes[0].set_title(f"Frame {i + 1} - Color")
        axes[0].axis("off")

        axes[1].imshow(frame_gris, cmap="gray")
        axes[1].set_title(f"Frame {i + 1} - Escala de Grises")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    captura_video.release()
    cv.destroyAllWindows()


def show_video(video_path: str):
    """
    Muestra un video cuadro a cuadro en una ventana emergente utilizando OpenCV.

    Parámetros:
    - video_path: Ruta al archivo de video que se desea visualizar.

    Comportamiento:
    - Se redimensiona cada frame del video a la mitad de su tamaño original para reducir el consumo de recursos.
    - Se muestra cada frame en una ventana OpenCV titulada 'Video'.
    - Se puede interrumpir la reproducción presionando la tecla 'q'.
    """
    # Abre el video
    captura_video = cv.VideoCapture(video_path)

    if not captura_video.isOpened():
        print("Error al abrir el archivo de video")
        return

    # Obtiene dimensiones del video
    frame_width = int(captura_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(captura_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    new_width = frame_width // 2
    new_height = frame_height // 2

    # Obtiene la tasa de cuadros por segundo
    fps = int(captura_video.get(cv.CAP_PROP_FPS))
    delay = int(600 / fps)

    while True:
        ret, frame = captura_video.read()
        if not ret:
            break

        # Redimensionar frame a la mitad
        frame_resized = cv.resize(frame, (new_width, new_height))

        # Mostrar frame
        cv.imshow("Video", frame_resized)

        # Salir con la tecla 'q'
        if cv.waitKey(delay) & 0xFF == ord("q"):
            break

    # Liberar recursos
    captura_video.release()
    cv.destroyAllWindows()
    cv.waitKey(1)


def measure_image_quality_fm_metric(image: np.ndarray, threshold_factor: int = 1000):
    """
    Calcula el valor de nitidez FM (Frequency Domain Image Blur Measure)
    basado en el paper "Image Sharpness Measure for Blurred Images in Frequency Domain".

    Parámetro:
        image: Imagen en escala de grises (2D)

    Retorna:
        float: Medida de nitidez (FM) en dominio de frecuencia
    """
    # Paso 1: Transformada de Fourier
    F = np.fft.fft2(image)

    # Paso 2: Centrar la transformada
    Fc = np.fft.fftshift(F)

    # Paso 3: Módulo del espectro
    AF = np.abs(Fc)

    # Paso 4: Máximo valor del espectro
    M = np.max(AF)

    # Paso 5: Umbral (threshold = M / 1000), contar valores mayores al umbral
    threshold = M / threshold_factor
    TH = np.sum(AF > threshold)

    # Paso 6: Calcular FM como TH dividido por el número total de píxeles
    FM = TH / (image.shape[0] * image.shape[1])

    return FM


def process_video(video_path: str) -> Tuple[list[dict[str, float]], list[np.ndarray]]:
    """
    Procesa un video cuadro a cuadro para calcular la métrica de enfoque (Focus Measure, FM)
    utilizando measure_image_quality_fm_metric.

    Parámetros:
    - video_path: Ruta al archivo de video que se desea procesar.

    Retorna:
    - fm_values (list of dict): Lista con diccionarios, cada uno contiene:
        - 'frame': índice del cuadro
        - 'fm': valor de la métrica de enfoque para ese cuadro
    - frames_processed (list of ndarray): Lista de los frames procesados (en escala de grises).

    Ejemplo:
    ```python
    fm_vals, frames = process_video("mi_video.mp4")
    ```
    """
    captura_video = cv.VideoCapture(video_path)
    fm_values = []
    frames_processed = []

    if not captura_video.isOpened():
        print("Error al abrir el archivo de video")
        return None, None

    frame_idx = 0
    while True:
        ret, frame = captura_video.read()
        if not ret:
            break

        # Conversión a escala de grises sin redimensionar
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calcular métrica de enfoque
        fm = measure_image_quality_fm_metric(frame_gray)

        # Guardar métricas y frame
        fm_values.append({"frame": frame_idx, "fm": fm})
        frames_processed.append(frame)

        frame_idx += 1

    captura_video.release()
    return fm_values, frames_processed


def export_and_plot_fm(fm_values, csv_output):
    df_fm = pd.DataFrame(fm_values)
    csv_path = csv_output + "/frecuencia_fm.csv"
    plot_path = csv_output + "/frecuencia_fm.png"

    df_fm.to_csv(csv_path, index=False)
    print(f"CSV exportado a: {csv_path}")

    plt.figure(figsize=(10, 4))
    plt.plot(df_fm["frame"], df_fm["fm"], label="FM por Frame")
    plt.xlabel("Frame")
    plt.ylabel("Métrica FM")
    plt.title("Medida de Nitidez en el Dominio de Frecuencia (FM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Gráfico guardado como: {plot_path}")
    plt.show()


def show_extreme_frames(fm_values, frames_processed, n=3):
    df_fm = pd.DataFrame(fm_values)
    df_sorted_asc = df_fm.sort_values("fm", ascending=True).reset_index(drop=True)
    df_sorted_desc = df_fm.sort_values("fm", ascending=False).reset_index(drop=True)

    low_indices = df_sorted_asc.loc[: n - 1, "frame"].values
    high_indices = df_sorted_desc.loc[: n - 1, "frame"].values

    print(f"Mostrando los {n} frames con menor FM:")
    for i, idx in enumerate(low_indices):
        show_image(
            frames_processed[idx],
            f'FM Bajo #{i+1} - Frame {idx} - FM={df_fm.loc[df_fm["frame"] == idx, "fm"].values[0]:.5f}',
        )

    print(f"Mostrando los {n} frames con mayor FM:")
    for i, idx in enumerate(high_indices):
        show_image(
            frames_processed[idx],
            f'FM Alto #{i+1} - Frame {idx} - FM={df_fm.loc[df_fm["frame"] == idx, "fm"].values[0]:.5f}',
        )


def measure_grid_focus_map(
    image: np.ndarray,
    n_rows: int,
    n_cols: int,
    *,
    threshold_factor: float = 1000.0,
) -> np.ndarray:
    """
    Calcula la métrica FM en una rejilla regular de N×M sobre una imagen en escala de grises.

    Parámetros
    ----------
    image : np.ndarray  # shape (H, W), dtype cualquiera
        Imagen en escala de grises.
    n_rows, n_cols : int
        Dimensiones de la rejilla (deben ser > 0).
    threshold_factor : float, opcional
        Factor divisor global usado dentro de `measure_image_quality_fm_metric`.

    Devuelve
    -------
    np.ndarray de forma (n_rows, n_cols) con dtype float32
        Valor de FM por celda.
    """

    if image.ndim != 2:
        raise ValueError("`image` must be 2-D grayscale")

    H, W = image.shape
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("`n_rows` and `n_cols` must be positive")

    # Edges so every pixel is used exactly once
    row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, W, n_cols + 1, dtype=int)

    fm_map = np.empty((n_rows, n_cols), dtype=np.float32)

    for i in range(n_rows):
        y0, y1 = row_edges[i], row_edges[i + 1]
        for j in range(n_cols):
            x0, x1 = col_edges[j], col_edges[j + 1]
            tile = image[y0:y1, x0:x1]
            fm_map[i, j] = measure_image_quality_fm_metric(
                tile, threshold_factor=threshold_factor
            )

    return fm_map
