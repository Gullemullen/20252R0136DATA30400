# 20252R0136DATA30400

Final Project for **DATA304: Big Data Analysis**
Hierarchical Multi-Label Text Classification

## Author Information
- Student ID: 2025952074
- Course: DATA304 – Big Data Analysis
- Kaggle Team Name: korea-dt-11
- GitHub ID: Gullemullen

This repository contains the implementation for the final project, including data exploration, silver label generation, model training, and prediction generation for Kaggle submission.

## Project Overview
The goal of this project is to perform **hierarchical multi-label classification** on unlabeled product reviews using a predefined product taxonomy and class-related keywords. No gold labels are provided; instead, silver labels are generated and refined to train classification models.

## Repository Structure
Each Jupyter notebook (`.ipynb`) is designed to run **independently** from top to bottom, assuming the directory structure is preserved and required files are available locally.

### Notebooks
1. **Input.ipynb**
   - Loads and explores the raw training and test review corpora
   - Examines class taxonomy structure and class keyword distributions
   - Performs basic text statistics and preprocessing analysis

2. **Representation & Silver Label Generation**
    - TF-IDF representations with cosine similarity between reviews and class keyword documents
    - Dense static embeddings via word co-occurrence matrix and Truncated SVD
    - Transformer-based review and class embeddings using a pretrained sentence encoder
    - Similarity-based silver label generation using top-k selection with confidence thresholds
    - Diagnostic analysis of confidence scores and label distributions

3. **Baseline Multi-Label Classifier.ipynb**
   - Trains a baseline multi-label classifier using the generated silver labels
   - Generates prediction files in the required Kaggle submission format

## Notes
- Random seeds are fixed for Python, NumPy, and PyTorch to minimize randomness.
- All experiments are executed using only the provided dataset and resources.
- Intermediate results (e.g., silver labels) are generated within the notebooks.
