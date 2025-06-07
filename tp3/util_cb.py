import cv2 as cv
import numpy as np
import supervision as sv

def load_and_preprocess_template(image_path, invert=False, equalize=True):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detectar fondo blanco para recortar
    _, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY_INV)
    x, y, w, h = cv.boundingRect(thresh)

    # Recorte del área útil
    cropped = gray[y:y + h, x:x + w]

    # Invertir si se solicita
    if invert:
        cropped = 255 - cropped

    # Ecualización (CLAHE)
    if equalize:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        template = clahe.apply(cropped)
    else:
        template = cropped

    # Devolver: imagen original, template procesado, y posición del recorte
    return img, template, (x, y, w, h)

def preprocess_target(path, invert=False, equalize=True, blur=False):
    """
    Carga y preprocesa una imagen destino (target) para matching.

    Args:
        path (str): Ruta a la imagen.
        invert (bool): Si se debe invertir la imagen.
        equalize (bool): Si se debe aplicar CLAHE para mejorar contraste.
        blur (bool): Si se debe aplicar un ligero desenfoque para reducir ruido.

    Returns:
        target_raw: Imagen original en escala de grises.
        target_proc: Imagen preprocesada para matching.
    """
    target_img = cv.imread(path)
    target_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
    assert target_gray is not None, f"Error al cargar imagen: {path}"

    target_proc = target_gray.copy()

    if invert:
        target_proc = 255 - target_proc

    if blur:
        target_proc = cv.GaussianBlur(target_proc, (3, 3), 0)

    if equalize:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_proc = clahe.apply(target_proc)

    return target_img, target_proc

def find_kp_and_des(img):
    sift = cv.SIFT_create()
    return sift.detectAndCompute(img, None)

def match_and_show(template_raw, template, template_crop_xy, target_raw, target, kp1, des1, kp2, des2):
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 4

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if M is not None:
            matchesMask = mask.ravel().tolist()
            good_inliers = [m for i, m in enumerate(good) if matchesMask[i]]

            # Proyectar área del template
            h_t, w_t = template.shape
            box = np.float32([[0, 0], [0, h_t], [w_t, h_t], [w_t, 0]]).reshape(-1, 1, 2)
            projected = cv.perspectiveTransform(box, M)

            # Dibujar polígono proyectado en la imagen color
            cv.polylines(target, [np.int32(projected)], True, (0, 255, 0), 3, cv.LINE_AA)
        else:
            print("No se pudo estimar homografía.")
            good_inliers = []
    else:
        print(f"Insuficientes matches válidos: {len(good)} / {MIN_MATCH_COUNT}")
        good_inliers = []

    # Dibujar matches
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=None,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_out = cv.drawMatches(template_raw, kp1, target_raw, kp2, good_inliers, None, **draw_params)

    # Mostrar con supervision
    sv.plot_image(img_out)

def offset_keypoints(kps, offset):
    # offset: (x, y)
    for kp in kps:
        kp.pt = (kp.pt[0] + offset[0], kp.pt[1] + offset[1])
    return kps

def match_and_show_2(template_raw, crop_xy, target_raw, kp1, des1, kp2, des2):
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 4

    # Aplica offset a keypoints del template para que estén en coords de template_raw
    kp1_offset = offset_keypoints(kp1, crop_xy)

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1_offset[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if M is not None:
            matchesMask = mask.ravel().tolist()
            good_inliers = [m for i, m in enumerate(good) if matchesMask[i]]

            # Proyectar área del template (en coords originales)
            h_t, w_t = template_raw.shape[:2]
            box = np.float32([[0, 0], [0, h_t], [w_t, h_t], [w_t, 0]]).reshape(-1, 1, 2)
            projected = cv.perspectiveTransform(box, M)

            # Dibujar polígono proyectado en la imagen color
            cv.polylines(target_raw, [np.int32(projected)], True, (0, 255, 0), 3, cv.LINE_AA)
        else:
            print("No se pudo estimar homografía.")
            good_inliers = []
            matchesMask = None
    else:
        print(f"Insuficientes matches válidos: {len(good)} / {MIN_MATCH_COUNT}")
        good_inliers = []
        matchesMask = None

    # Ahora dibujar matches entre template_raw y target_color
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_out = cv.drawMatches(template_raw, kp1_offset, target_raw, kp2, good, None, matchesMask=matchesMask,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    sv.plot_image(img_out)
