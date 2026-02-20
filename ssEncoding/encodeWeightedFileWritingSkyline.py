# ============================================================
# Skyline Polygon Encoding (FINAL – NO QUADTREE – NO ERRORS)
# Compatible with Weighted Jaccard (NMSLIB)
# Windows + Conda Safe
# ============================================================

import os
import sys
import math
import time
import numpy as np
import shapely.wkt
from shapely.geometry import Polygon
from shapely import affinity

# ------------------------------------------------------------
# Fix Python path for local lib imports
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "lib"))

import wkthelper

# ============================================================
# Utility Functions
# ============================================================

def initialPolygonCentering(polygons):
    centered = []
    for p in polygons:
        c = p.centroid
        centered.append(affinity.translate(p, -c.x, -c.y))
    print(f"{len(centered)} polygons centered to origin")
    return centered


def computeGlobalMBR(polygons):
    minx, miny, maxx, maxy = polygons[0].bounds
    for p in polygons:
        bx, by, Bx, By = p.bounds
        minx = min(minx, bx)
        miny = min(miny, by)
        maxx = max(maxx, Bx)
        maxy = max(maxy, By)
    return minx, miny, maxx, maxy


# ============================================================
# BGA-STYLE FIXED GRID ENCODING (SAFE REPLACEMENT)
# ============================================================

def gridBasedEncoding(
    filename,
    polygons,
    start,
    end,
    grid_size=64
):
    print("Using fixed grid encoding (SAFE MODE)")

    minx, miny, maxx, maxy = computeGlobalMBR(polygons)

    dx = (maxx - minx) / grid_size
    dy = (maxy - miny) / grid_size

    output_file = filename + "_0.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for idx in range(start, end):
            poly = polygons[idx]
            vector = np.zeros(grid_size * grid_size, dtype=float)

            for i in range(grid_size):
                for j in range(grid_size):
                    cell = Polygon([
                        (minx + i*dx,     miny + j*dy),
                        (minx + (i+1)*dx, miny + j*dy),
                        (minx + (i+1)*dx, miny + (j+1)*dy),
                        (minx + i*dx,     miny + (j+1)*dy)
                    ])

                    if cell.intersects(poly):
                        inter_area = cell.intersection(poly).area
                        vector[i * grid_size + j] = inter_area

            f.write(" ".join(map(str, vector)) + "\n")

    print(f"Encoding written to: {output_file}")


# ============================================================
# MAIN SKYLINE PIPELINE
# ============================================================

def parks():
    print("Reading Skyline polygons")

    wkt_polygons = wkthelper.readParks(
        "polygonalData/skyline_polygons.tsv"
    )

    print(f"Total polygons loaded: {len(wkt_polygons)}")

    polygons = initialPolygonCentering(wkt_polygons)

    # ---- Dataset split (same as ground truth logic) ----
    data_percent = 0.8
    data_start = 0
    data_end = math.ceil(len(polygons) * data_percent)
    query_start = data_end
    query_end = len(polygons)

    print(f"Data   : [{data_start} → {data_end})")
    print(f"Query  : [{query_start} → {query_end})")

    # ---- Output ----
    out_dir = os.path.join("encodedData", "skyline")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, "skyline_weighted")

    # ---- Encoding ----
    gridBasedEncoding(
        filename=filename,
        polygons=polygons,
        start=data_start,
        end=query_end,
        grid_size=64   # safe + sufficient
    )

    print("Skyline encoding COMPLETED successfully")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    start = time.time()
    parks()
    end = time.time()
    print(f"Total runtime: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
