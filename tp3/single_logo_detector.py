import cv2 as cv
import numpy as np
import supervision as sv


class SingleLogoDetector:
    """
    Detector de logotipo basado en coincidencia de características (SIFT) y homografía.

    Atributos:
        template_path (str): Ruta a la imagen del logotipo (template).
        target_path (str): Ruta a la imagen objetivo donde se busca el logotipo.
        best_result: Resultado con mejor coincidencia tras ejecutar `run()`.
    """

    def __init__(self, template_path, target_path):
        self.template_path = template_path
        self.target_path = target_path
        self.best_result = None

    def load_and_preprocess_template(self, invert=False, equalize=True):
        """
        Carga la imagen template, la convierte a escala de grises y la recorta al contenido.

        Args:
            invert (bool): Si se debe invertir (negativo) la imagen.
            equalize (bool): Si se debe aplicar ecualización CLAHE.

        Returns:
            tuple: (imagen original, imagen procesada, offset de recorte)
        """
        img = cv.imread(self.template_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY_INV)
        x, y, w, h = cv.boundingRect(thresh)
        cropped = gray[y:y + h, x:x + w]
        if invert:
            cropped = 255 - cropped
        if equalize:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            template = clahe.apply(cropped)
        else:
            template = cropped
        return img, template, (x, y)

    def preprocess_target(self):
        """
        Preprocesa la imagen objetivo.

        Returns:
            tuple: (imagen original, imagen procesada)
        """
        target_img = cv.imread(self.target_path)
        gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
        target_proc = gray.copy()
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_proc = clahe.apply(target_proc)
        return target_img, target_proc

    def find_kp_and_des(self, image):
        """
        Detecta keypoints y descriptores con SIFT.

        Args:
            image (np.ndarray): Imagen en escala de grises.

        Returns:
            tuple: (keypoints, descriptores)
        """
        sift = cv.SIFT_create()
        return sift.detectAndCompute(image, None)

    def offset_keypoints(self, kps, offset):
        """
        Ajusta la posición de keypoints según el offset del recorte.

        Args:
            kps (list): Lista de keypoints.
            offset (tuple): (x, y) desplazamiento.

        Returns:
            list: Keypoints ajustados.
        """
        x, y = offset
        for kp in kps:
            kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        return kps

    def match_keypoints(self, des1, des2):
        """
        Realiza la coincidencia de descriptores con el ratio test de Lowe.

        Args:
            des1, des2: Descriptores de template y target.

        Returns:
            tuple: (matches buenos, todos los matches)
        """
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return good, matches

    def compute_homography(self, kp1, kp2, matches):
        """
        Calcula la homografía entre puntos coincidentes.

        Args:
            kp1, kp2: Keypoints de template y target.
            matches: Matches buenos.

        Returns:
            tuple: (matriz de homografía, máscara de inliers)
        """
        if len(matches) < 4:
            return None, None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        return M, mask

    def compute_confidence(self, matches, mask):
        """
        Calcula la confianza como el ratio de inliers vs matches totales.

        Args:
            matches: Matches buenos.
            mask: Máscara de inliers.

        Returns:
            float: Confianza (0 a 1)
        """
        if mask is None or len(matches) == 0:
            return 0.0
        num_inliers = np.sum(mask)
        return num_inliers / len(matches)

    def draw_result(self, template_raw, kp1, target_raw, kp2, matches, mask, M):
        """
        Dibuja el resultado: polígono proyectado y confianza sobre la imagen.

        Args:
            template_raw: Imagen original del template.
            kp1, kp2: Keypoints de ambas imágenes.
            target_raw: Imagen original del target.
            matches: Matches buenos.
            mask: Máscara de inliers.
            M: Matriz de homografía.
        """
        confidence = self.compute_confidence(matches, mask)

        if M is not None:
            h_t, w_t = template_raw.shape[:2]
            box = np.float32([[0, 0], [0, h_t], [w_t, h_t], [w_t, 0]]).reshape(-1, 1, 2)
            projected = cv.perspectiveTransform(box, M)

            # Dibujar polígono verde
            cv.polylines(target_raw, [np.int32(projected)], True, (0, 255, 0), 3, cv.LINE_AA)

            # Obtener vértice superior derecho
            top_right = projected[3][0]

            # Preparar texto de confianza
            confidence_text = f"{confidence:.0%}"
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv.getTextSize(confidence_text, font, font_scale, thickness)

            text_x = int(top_right[0] - text_width)
            text_y = int(top_right[1] - 10)

            # Dibujar fondo verde
            cv.rectangle(
                target_raw,
                (text_x - 2, text_y - text_height - baseline - 2),
                (text_x + text_width + 2, text_y + 2),
                (0, 255, 0),
                -1
            )

            # Dibujar texto negro
            cv.putText(
                target_raw,
                confidence_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv.LINE_AA
            )

        matchesMask = mask.ravel().tolist() if mask is not None else None
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        result = cv.drawMatches(template_raw, kp1, target_raw, kp2, matches, None, **draw_params)
        sv.plot_image(result)

    def run(self):
        """
        Ejecuta todo el pipeline de detección:
        - Prueba con template normal e invertido.
        - Busca keypoints y matches.
        - Calcula homografía.
        - Dibuja resultado con confianza.
        """
        target_raw, target_proc = self.preprocess_target()

        best = {'matches': [], 'variant': None, 'data': None}
        for invert in [False, True]:
            template_raw, template_proc, offset = self.load_and_preprocess_template(invert=invert)
            kp1, des1 = self.find_kp_and_des(template_proc)
            kp2, des2 = self.find_kp_and_des(target_proc)
            matches, all_matches = self.match_keypoints(des1, des2)

            if len(matches) > len(best['matches']):
                kp1_offset = self.offset_keypoints(kp1, offset)
                best = {
                    'matches': matches,
                    'variant': 'invertido' if invert else 'normal',
                    'data': (template_raw, kp1_offset, des1, kp2, des2, target_raw)
                }

        if not best['matches']:
            print("No se encontraron matches válidos en ninguna variante.")
            return

        print(f"Mejor variante: {best['variant']} con {len(best['matches'])} matches.")
        template_raw, kp1, des1, kp2, des2, target_raw = best['data']
        M, mask = self.compute_homography(kp1, kp2, best['matches'])
        self.draw_result(template_raw, kp1, target_raw, kp2, best['matches'], mask, M)
