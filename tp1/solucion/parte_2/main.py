from util import load_image_cv2,read_image_cv2,make_multiple_hist,LoadMode


path = "material/parte2/img1_tp.png"

image_gray = load_image_cv2(path,mode=LoadMode.GRAY)
read_image_cv2(image_gray,"imagen gris")

make_multiple_hist(image_gray, bins_list=[10, 30, 60], output_dir="material/parte2/solucion/histograms")