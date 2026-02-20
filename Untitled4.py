import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

main_folder = "images"

valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
image_paths = [p for p in Path(main_folder).rglob("*") if p.suffix.lower() in valid_ext]

print(f"Total images found: {len(image_paths)}")

def detect_top_skyline(img):
    h, w = img.shape[:2]
    max_w = 1200
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    skyline_x = []
    skyline_y = []

    for x in range(edges.shape[1]):
        col = edges[:, x]
        idx = np.where(col > 0)[0]
        if len(idx):
            skyline_x.append(x)
            skyline_y.append(idx[0])

    if len(skyline_x) < 3:
        return None, None, img

    return skyline_x, skyline_y, img

def create_polygon_from_skyline(skyline_x, skyline_y, img):
    h, w = img.shape[:2]

    skyline_y = np.array(skyline_y, dtype=int)
    skyline_y = cv2.blur(skyline_y.reshape(1, -1), (1, 9)).flatten().astype(int)

    max_top = max(skyline_y)
    y_bottom = min(max_top + 20, h - 1)

    x_left = skyline_x[0]
    x_right = skyline_x[-1]

    polygon = [(x_left, y_bottom)]
    polygon.extend(zip(skyline_x, skyline_y))
    polygon.append((x_right, y_bottom))

    return polygon

# --------- EXPORT TO WKT (IMPORTANT PART) ----------
wkt_out = Path("ssEncoding/polygonalData/skyline_polygons.tsv")
wkt_out.parent.mkdir(parents=True, exist_ok=True)

with open(wkt_out, "w") as f:
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        skyline_x, skyline_y, resized_img = detect_top_skyline(img)

        if skyline_x is None:
            continue

        polygon = create_polygon_from_skyline(skyline_x, skyline_y, resized_img)
        poly = Polygon(polygon)

        if poly.is_valid:
            f.write(poly.wkt + "\n")

print(" WKT polygons saved to:", wkt_out)
