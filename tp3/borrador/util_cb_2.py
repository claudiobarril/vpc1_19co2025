import cv2 as cv
import numpy as np
import supervision as sv

# --------------------- CARGA Y PREPROCESADO ---------------------

def load_and_preprocess_template(image_path, invert=False, equalize=True):
    img = cv.imread(image_path)
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

def preprocess_target(image_path, invert=False, equalize=True, blur=False):
    target_img = cv.imread(image_path)
    gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
    target_proc = gray.copy()
    if invert:
        target_proc = 255 - target_proc
    if blur:
        target_proc = cv.GaussianBlur(target_proc, (3, 3), 0)
    if equalize:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_proc = clahe.apply(target_proc)
    return target_img, target_proc

# --------------------- KEYPOINTS Y DESCRIPTORES ---------------------

def find_kp_and_des(image):
    sift = cv.SIFT_create()
    return sift.detectAndCompute(image, None)

def offset_keypoints(kps, offset):
    x, y = offset
    for kp in kps:
        kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
    return kps

# --------------------- MATCHING Y HOMOGRAFÍA ---------------------

def match_keypoints(des1, des2):
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good, matches

def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return M, mask

def draw_result(template_raw, kp1, target_raw, kp2, matches, mask, M):
    confidence = compute_confidence(matches, mask)

    if M is not None:
        h_t, w_t = template_raw.shape[:2]
        box = np.float32([[0, 0], [0, h_t], [w_t, h_t], [w_t, 0]]).reshape(-1, 1, 2)
        projected = cv.perspectiveTransform(box, M)

        # Dibujar polígono verde
        cv.polylines(target_raw, [np.int32(projected)], True, (0, 255, 0), 3, cv.LINE_AA)

        # Obtener vértice superior derecho
        top_right = projected[3][0]

        # Preparar texto de confianza
        text = f"{confidence:.2f}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Obtener tamaño del texto
        (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, thickness)

        # Calcular punto inicial: que el extremo derecho del texto coincida con top_right
        text_x = int(top_right[0] - text_width)
        text_y = int(top_right[1] - 10)  # un poco por encima

        # Dibujar texto en verde
        cv.putText(target_raw, text, (text_x, text_y), font,
                   font_scale, (0, 255, 0), thickness, cv.LINE_AA)

    matchesMask = mask.ravel().tolist() if mask is not None else None
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    result = cv.drawMatches(template_raw, kp1, target_raw, kp2, matches, None, **draw_params)
    sv.plot_image(result)

def compute_confidence(matches, mask):
    if mask is None or len(matches) == 0:
        return 0.0
    num_inliers = np.sum(mask)
    return num_inliers / len(matches)

# --------------------- FUNCIÓN PRINCIPAL ---------------------

def run_matching_pipeline(template_path, target_path):
    target_raw, target_proc = preprocess_target(target_path)

    # Probar ambas variantes del template (invertida y no invertida)
    best = {'matches': [], 'variant': None, 'data': None}
    for invert in [False, True]:
        template_raw, template_proc, offset = load_and_preprocess_template(template_path, invert=invert)
        kp1, des1 = find_kp_and_des(template_proc)
        kp2, des2 = find_kp_and_des(target_proc)
        matches, all_matches = match_keypoints(des1, des2)

        if len(matches) > len(best['matches']):
            kp1_offset = offset_keypoints(kp1, offset)
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
    M, mask = compute_homography(kp1, kp2, best['matches'])
    draw_result(template_raw, kp1, target_raw, kp2, best['matches'], mask, M)
