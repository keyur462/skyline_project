# 🏙️ Skyline Polygon Similarity Search

> A high-performance spatial similarity search pipeline for Skyline polygons — combining geometric encoding, Weighted Jaccard similarity, and HNSW indexing to find the most shape-similar polygons at scale.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![NMSLIB](https://img.shields.io/badge/NMSLIB-HNSW-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 What This Project Does

Given a set of **Skyline polygons** (city skylines represented as geometric shapes), this project:

1. **Reads** raw polygon data in WKT format
2. **Generates** ground truth similarity using brute-force shape comparison
3. **Encodes** polygons into weighted numeric feature vectors via Grid / Quad-tree spatial partitioning
4. **Indexes** vectors using NMSLIB's HNSW approximate nearest neighbor index
5. **Queries** the index for the most shape-similar polygons (kNN search)
6. **Evaluates** quality using Recall@K and visualizes results

---

## 📁 Project Structure

```
skyline_project/
│
├── ssEncoding/                          ⭐ MAIN PROJECT FOLDER
│   │
│   ├── polygonalData/                   # Raw WKT polygon input data
│   ├── lib/                             # Core reusable modules
│   │   ├── wkthelper.py                 # WKT polygon parser
│   │   ├── quadtree.py                  # Quad-tree spatial partitioning
│   │   └── grid.py                      # Uniform grid encoding
│   │
│   ├── groundtruth/                     # Brute-force ground truth similarity maps
│   ├── groundtruthLog/                  # Logs from ground truth generation
│   ├── encodedData/                     # Numeric polygon encodings for NMSLIB
│   ├── nmslib/                          # NMSLIB source (local build)
│   ├── results_skyline/                 # Final recall results & summaries
│   │
│   ├── groundtruthWritingSkyline.py     # Generates ground truth similarity
│   ├── encodeWeightedFileWritingSkyline.py  # Encodes polygons → weighted vectors
│   ├── polygonSimilaritySearch.ipynb    # Main experiment notebook (skyline)
│   ├── polygonSimilaritySearch1.ipynb   # Alternate/variant notebook
│   ├── polygonSimilaritySearch-parks50k.ipynb  # Scalability test (50k polygons)
│   └── install_WeightedJaccard_NMSLIB/  # NMSLIB install guide (Windows)
│
├── output/                              # Experimental/debug outputs
├── output_opencv_skyline/               # OpenCV-based skyline visuals
└── README.md
```

---

## ⚙️ End-to-End Pipeline

```
Raw Polygons (WKT)
      ↓
Ground Truth Generation  →  groundtruthWritingSkyline.py
      ↓
Polygon Encoding         →  encodeWeightedFileWritingSkyline.py
      ↓
NMSLIB Index + kNN Query →  polygonSimilaritySearch.ipynb
      ↓
Recall@K Evaluation + Plots
      ↓
Results saved to results_skyline/
```

---

## 🧠 Core Concepts

**Weighted Jaccard Similarity** — measures how similar two polygon encodings are by comparing their spatial cell overlap, weighted by intersection area. Far more accurate for shapes than standard Jaccard.

**Grid / Quad-tree Encoding** — each polygon is projected onto a spatial grid or quad-tree. Cells the polygon overlaps become features, with weights proportional to the area of intersection.

**HNSW via NMSLIB** — Hierarchical Navigable Small World graph index enables fast approximate nearest-neighbor search, making kNN queries scalable to large datasets.

**Recall@K** — measures how many of the true top-K similar polygons were correctly found by the approximate search. Higher is better.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| NMSLIB | HNSW approximate nearest neighbor index |
| Weighted Jaccard | Polygon similarity metric |
| Shapely | Geometric operations |
| NumPy | Vector math |
| Matplotlib | Recall@K visualization |
| Jupyter Notebook | Experiments & evaluation |
| OpenCV | Skyline image/contour processing |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/skyline-polygon-similarity.git
cd skyline-polygon-similarity/ssEncoding
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install NMSLIB with Weighted Jaccard Support

> See `install_WeightedJaccard_NMSLIB/` for detailed instructions.
> Windows users need to compile NMSLIB from source (guide included).

```bash
pip install nmslib
```

### 4. Generate Ground Truth

```bash
python groundtruthWritingSkyline.py
```

### 5. Encode Polygons

```bash
python encodeWeightedFileWritingSkyline.py
```

### 6. Run the Experiment

Open and run the main notebook:

```bash
jupyter notebook polygonSimilaritySearch.ipynb
```

---

## 📊 Results

- ✅ Weighted Jaccard similarity fully implemented and verified
- ✅ HNSW index built and queried successfully
- ✅ Recall@K computed across multiple K values
- ✅ Results saved to `results_skyline/`
- ✅ Scalability tested on Parks dataset (50,000 polygons)

---

## 📂 Datasets

| Dataset | Size | File |
|---------|------|------|
| Skyline Polygons | 120 polygons | `polygonalData/skyline_polygons.tsv` |
| Parks Dataset | 50,000 polygons | Used in `polygonSimilaritySearch-parks50k.ipynb` |

All data is in **WKT (Well-Known Text)** format.

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- [NMSLIB](https://github.com/nmslib/nmslib) — for the efficient HNSW similarity search library
- Shapely & NumPy open-source communities
