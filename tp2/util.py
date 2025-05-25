import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def show_image(image, title=''):
    """
    Muestra una imagen en una celda usando matplotlib.

    Parámetros:
    - imagen (np.ndarray): Imagen a mostrar, puede estar en escala de grises (2D) o en color (3D en BGR).
    - title (str): Título a mostrar sobre la imagen.

    Nota:
    Si la imagen está en formato BGR (como es usual con OpenCV), se convierte a RGB para mostrar correctamente los colores.
    """
    plt.figure(figsize=(5, 4))
    if len(image.shape) == 3:
        imagen_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(imagen_rgb)
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_n_frames(video_path, n):
    """
    Lee y muestra los primeros 'n' frames de un video, en versión original (color)
    y escala de grises, una al lado de la otra.

    Parámetros:
    - video_path (str): Ruta al archivo de video.
    - n (int): Número de frames a mostrar.

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
        axes[0].set_title(f'Frame {i + 1} - Color')
        axes[0].axis('off')

        axes[1].imshow(frame_gris, cmap='gray')
        axes[1].set_title(f'Frame {i + 1} - Escala de Grises')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    captura_video.release()
    cv.destroyAllWindows()


def show_video(video_path):
    """
    Muestra un video cuadro a cuadro en una ventana emergente utilizando OpenCV.

    Parámetros:
    - video_path (str): Ruta al archivo de video que se desea visualizar.

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
        cv.imshow('Video', frame_resized)

        # Salir con la tecla 'q'
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break

    # Liberar recursos
    captura_video.release()
    cv.destroyAllWindows()
    cv.waitKey(1)


def measure_image_quality_fm_metric(image):
    """
    Calcula el valor de nitidez FM (Frequency Domain Image Blur Measure)
    basado en el paper "Image Sharpness Measure for Blurred Images in Frequency Domain".

    Parámetro:
        image (np.ndarray): Imagen en escala de grises (2D)

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
    threshold = M / 1000.0
    TH = np.sum(AF > threshold)

    # Paso 6: Calcular FM como TH dividido por el número total de píxeles
    FM = TH / (image.shape[0] * image.shape[1])

    return FM


def process_video(video_path):
    """
    Procesa un video cuadro a cuadro para calcular la métrica de enfoque (Focus Measure, FM)
    utilizando measure_image_quality_fm_metric.

    Parámetros:
    - video_path (str): Ruta al archivo de video que se desea procesar.

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
        fm_values.append({'frame': frame_idx, 'fm': fm})
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
    plt.plot(df_fm['frame'], df_fm['fm'], label='FM por Frame')
    plt.xlabel('Frame')
    plt.ylabel('Métrica FM')
    plt.title('Medida de Nitidez en el Dominio de Frecuencia (FM)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Gráfico guardado como: {plot_path}")
    plt.show()


def show_extreme_frames(fm_values, frames_processed, n=3):
    df_fm = pd.DataFrame(fm_values)
    df_sorted_asc = df_fm.sort_values('fm', ascending=True).reset_index(drop=True)
    df_sorted_desc = df_fm.sort_values('fm', ascending=False).reset_index(drop=True)

    low_indices = df_sorted_asc.loc[:n-1, 'frame'].values
    high_indices = df_sorted_desc.loc[:n-1, 'frame'].values

    print(f"Mostrando los {n} frames con menor FM:")
    for i, idx in enumerate(low_indices):
        show_image(frames_processed[idx], f'FM Bajo #{i+1} - Frame {idx} - FM={df_fm.loc[df_fm["frame"] == idx, "fm"].values[0]:.5f}')

    print(f"Mostrando los {n} frames con mayor FM:")
    for i, idx in enumerate(high_indices):
        show_image(frames_processed[idx], f'FM Alto #{i+1} - Frame {idx} - FM={df_fm.loc[df_fm["frame"] == idx, "fm"].values[0]:.5f}')