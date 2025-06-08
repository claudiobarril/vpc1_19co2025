import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional


class MatchMethods:
    TM_CCOEFF = cv.TM_CCOEFF
    TM_CCOEFF_NORMED = cv.TM_CCOEFF_NORMED
    TM_CCORR = cv.TM_CCORR
    TM_CCORR_NORMED = cv.TM_CCORR_NORMED
    TM_SQDIFF = cv.TM_SQDIFF
    TM_SQDIFF_NORMED = cv.TM_SQDIFF_NORMED
    ALL = [
        TM_CCOEFF,
        TM_CCOEFF_NORMED,
        TM_CCORR,
        TM_CCORR_NORMED,
        TM_SQDIFF,
        TM_SQDIFF_NORMED,
    ]


def load_gray(path: str) -> np.ndarray:
    """Lee una imagen y la devuelve en escala de grises.."""
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def verify_template_size(img: np.ndarray, tmpl: np.ndarray) -> None:
    if tmpl.shape[0] > img.shape[0] or tmpl.shape[1] > img.shape[1]:
        raise ValueError(
            f"Template {tmpl.shape[::-1]} larger than image {img.shape[::-1]}"
        )


def preprocess_canny_edges(gray: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento con Canny: detecta bordes fuertes y los dilata levemente.
    """
    edges = cv.Canny(gray, 50, 150)
    return cv.dilate(edges, None, iterations=1)


def piramides(img_gray, tmpl_gray, scales, method=MatchMethods.TM_CCOEFF_NORMED):
    best = (-1, None)
    img_e = preprocess_canny_edges(img_gray)
    for s in scales:
        th, tw = int(tmpl_gray.shape[0] * s), int(tmpl_gray.shape[1] * s)
        if th < 10 or tw < 10 or th > img_gray.shape[0] or tw > img_gray.shape[1]:
            continue
        tmpl_e = preprocess_canny_edges(cv.resize(tmpl_gray, (tw, th), cv.INTER_AREA))
        res = cv.matchTemplate(img_e, tmpl_e, method)
        _, max_val, _, max_loc = cv.minMaxLoc(res)
        if max_val > best[0]:
            best = (max_val, (*max_loc, tw, th, max_val))
    return best[1]


def match_template(
    img_gray,
    tmpl_gray,
    thr: float = 0.8,
    method: int = MatchMethods.TM_CCOEFF_NORMED,
    scales=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
):
    """
    Devuelve una lista de detecciones [(x, y, puntuación)] donde la puntuación ≥ umbral,
    buscando el template en múltiples escalas reducidas para que siempre encaje.
    """

    det = piramides(img_gray, tmpl_gray, scales, method)
    if det and det[-1] >= thr:
        x, y, w, h, score = det
        return [(x, y, w, h, score)]
    return []


def draw_boxes(
    bgr: np.ndarray, detections, w: int, h: int, color=(0, 255, 0), thickness=2
):
    for x, y, score in detections:
        cv.rectangle(bgr, (x, y), (x + w, y + h), color, thickness)
        cv.putText(
            bgr,
            f"{score:.2f}",
            (x, y - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv.LINE_AA,
        )
    return bgr


def locate_logo(
    image_path: str, template_path: str, thr: float = 0.8, show: bool = True
):
    img_gray = load_gray(image_path)
    tmpl_gray = load_gray(template_path)

    detections = match_template(img_gray, tmpl_gray, thr)

    img_bgr = cv.imread(image_path)
    for x, y, w, h, score in detections:
        cv.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(
            img_bgr,
            f"{score:.2f}",
            (x, y - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

    if show:
        plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return detections


def run_folder(images_dir: str, template_path: str, thr: float = 0.8):
    """
    Recorre el directorio, dibuja los matches sobre cada imagen.
    """
    dir_path = Path(images_dir)
    for img_path in sorted(dir_path.glob("*.png")):
        print(f"\n{img_path.name}:")
        dets = locate_logo(str(img_path), template_path, thr, show=True)
        if not dets:
            print("  (no hits)")
        else:
            for x, y, w, h, s in dets:
                print(f"  ↳ at ({x},{y})  size={w}×{h}  score={s:.2f}")


