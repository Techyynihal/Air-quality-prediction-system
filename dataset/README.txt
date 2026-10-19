PLACE THE DATASET HERE
======================

Project: Air Quality Prediction Using Deep Learning (Image Classification)
Dataset: Air Pollution Image Dataset from India and Nepal
Source : https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal

HOW TO SET UP
-------------
1. Download the dataset from the Kaggle link above.
2. Extract it so that this `dataset/` directory contains the dataset's
   image folders (and any CSV/metadata files) as downloaded.

   The pipeline AUTO-DISCOVERS structure. It supports BOTH:
     - folder-based labels   (class name = sub-folder name)
     - CSV / metadata labels  (a column maps image -> label)

   You do NOT need to rename or reorganise anything by hand. Just extract
   the dataset into this folder. Example of what it may look like:

       dataset/
       ├── <Class A>/  *.jpg
       ├── <Class B>/  *.jpg
       ├── ...
       └── <optional>.csv

3. From the project root, run the environment + dataset audit first:

       python run_all.py --stage audit

   Then follow README.md for the full pipeline.

NOTES
-----
- Nothing in this project assumes the number of images, class names,
  class balance, resolution, or CSV columns. Everything is measured from
  the real files during the audit stage (see results/audit/).
- The test split is created once and never used for tuning. Real metrics
  only appear in results/ after you actually run training + evaluation.
