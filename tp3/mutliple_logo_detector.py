import cv2 as cv
import numpy as np

from typing import List, Tuple, Optional


class MultipleLogoDetector:
    """
    Detector de múltiples logos usando template matching multi-escala
    """

    def __init__(self, template_path: str):
        """
        Inicializa el detector con el template.

        Args:
            template_path: Ruta al template del logo
        """
        self.template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"No se pudo cargar el template desde {template_path}")

        # Usamos canny
        self.template_edges = cv.Canny(self.template, 120, 240, L2gradient=True)

        # Parámetros default
        self.min_scale_factor = 0.5
        self.max_scale_factor = 4.0
        self.scale_steps = 50
        self.confidence_threshold = 0.4
        self.nms_iou_threshold = 0.3

    def _generate_scales(self, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Genera escalas adaptativas según el tamaño de la imagen y template.

        Args:
            img_shape: Forma de la imagen (height, width)

        Returns:
            Array de escalas a evaluar
        """

        min_scale = (
            max(
                self.template.shape[0] / img_shape[0],
                self.template.shape[1] / img_shape[1],
            )
            * self.min_scale_factor
        )

        scales = np.logspace(
            np.log10(min_scale), np.log10(self.max_scale_factor), self.scale_steps
        )

        return scales

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: Imagen en escala de grises

        Returns:
            Imagen preprocesada
        """
        # Aplicar suavizado Gaussiano ligero para reducir ruido
        smoothed = cv.GaussianBlur(image, (3, 3), 0.5)

        # Detección de bordes con Canny usando parámetros similares al template
        edges = cv.Canny(smoothed, 100, 200, L2gradient=True)

        return edges

    @staticmethod
    def _calculate_iou(
            box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calcula Intersection over Union entre dos bounding boxes.

        Args:
            box1, box2: Tuplas (x1, y1, x2, y2) representando las cajas

        Returns:
            Valor IoU entre 0 y 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def _non_maximum_suppression(
        self,
        detections: List[Tuple[float, int, int, int, int]],
        iou_threshold: float = None,
    ) -> List[Tuple[float, int, int, int, int]]:
        """
        Aplica Non-Maximum Suppression para eliminar detecciones redundantes.

        Args:
            detections: Lista de (confidence, x1, y1, x2, y2)
            iou_threshold: Umbral IoU para suprimir detecciones

        Returns:
            Lista filtrada de detecciones
        """
        if not detections:
            return []

        if iou_threshold is None:
            iou_threshold = self.nms_iou_threshold

        # Ordenar por confianza descendente
        detections = sorted(detections, key=lambda x: x[0], reverse=True)

        filtered_detections = []

        while detections:
            # Tomar la detección con mayor confianza
            current = detections.pop(0)
            filtered_detections.append(current)

            # Filtrar detecciones con IoU alto respecto a la actual
            remaining = []
            for detection in detections:
                current_box = current[1:5]
                detection_box = detection[1:5]

                if self._calculate_iou(current_box, detection_box) < iou_threshold:
                    remaining.append(detection)

            detections = remaining

        return filtered_detections

    def detect_logos(
        self, image: np.ndarray, confidence_threshold: Optional[float] = None
    ) -> List[Tuple[float, int, int, int, int]]:
        """
        Detecta múltiples logos en la imagen usando template matching multi-escala.

        Args:
            image: Imagen donde buscar logos
            confidence_threshold: Umbral de confianza personalizado

        Returns:
            Lista de detecciones (confidence, x1, y1, x2, y2)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        # Pre-procesar imagen
        processed_image = self._preprocess_image(gray_image)

        # Generar escalas adaptativas
        scales = self._generate_scales(processed_image.shape)

        all_detections = []
        template_h, template_w = self.template_edges.shape

        # Búsqueda en múltiples escalas
        for scale in scales:
            # Redimensionar imagen
            scaled_height = int(processed_image.shape[0] * scale)
            scaled_width = int(processed_image.shape[1] * scale)

            if scaled_height < template_h or scaled_width < template_w:
                continue

            scaled_image = cv.resize(
                processed_image,
                (scaled_width, scaled_height),
                interpolation=cv.INTER_LINEAR,
            )

            # Template matching con correlación normalizada
            result = cv.matchTemplate(
                scaled_image, self.template_edges, cv.TM_CCOEFF_NORMED
            )

            # Encontrar ubicaciones que superan el umbral
            locations = np.where(result >= confidence_threshold)

            # Convertir ubicaciones a detecciones
            for pt in zip(*locations[::-1]):  # Intercambiar x,y
                # Calcular coordenadas en imagen original
                x1 = int(pt[0] / scale)
                y1 = int(pt[1] / scale)
                x2 = int((pt[0] + template_w) / scale)
                y2 = int((pt[1] + template_h) / scale)

                # Obtener confianza
                confidence = result[pt[1], pt[0]]

                all_detections.append((confidence, x1, y1, x2, y2))

        # Aplicar Non-Maximum Suppression
        filtered_detections = self._non_maximum_suppression(all_detections)

        return filtered_detections

    @staticmethod
    def visualize_detections(
            image: np.ndarray,
        detections: List[Tuple[float, int, int, int, int]],
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Visualiza las detecciones en la imagen con bounding boxes y confianza.

        Args:
            image: Imagen original
            detections: Lista de detecciones
            show_confidence: Si mostrar el porcentaje de confianza

        Returns:
            Imagen con las detecciones dibujadas
        """
        result_image = image.copy()

        # Convertir a RGB si está en BGR
        if len(result_image.shape) == 3:
            result_image = cv.cvtColor(result_image, cv.COLOR_BGR2RGB)

        for detection in detections:
            confidence, x1, y1, x2, y2 = detection

            # Dibujar bounding box
            cv.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Añadir texto con confianza si se solicita
            if show_confidence:
                confidence_text = f"{confidence:.0%}"
                font_scale = 0.6
                thickness = 2

                # Calcular tamaño del texto
                (text_width, text_height), baseline = cv.getTextSize(
                    confidence_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                # Dibujar fondo para el texto
                cv.rectangle(
                    result_image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width + 5, y1),
                    (0, 255, 0),
                    -1,
                )

                # Dibujar texto
                cv.putText(
                    result_image,
                    confidence_text,
                    (x1 + 2, y1 - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                )

        return result_image
