from util import run_folder

TEMPLATE = r"tp3\material\template\pattern.png"
IMAGES_DIR = r"tp3\material\images"

run_folder(IMAGES_DIR, TEMPLATE, thr=0.23)