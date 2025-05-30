# EEG Noise Classification using Topological Data Analysis (TDA)

## Overview

This project explores the use of **Topological Data Analysis (TDA)** for extracting robust features from EEG signals under various noise conditions. Using persistence images derived from persistence diagrams, we build classifiers to identify different noise types affecting EEG data.

The main goal is to demonstrate the effectiveness of TDA-based features combined with classical machine learning models such as **Random Forest** and **K-Nearest Neighbors (KNN)** in classifying noise artifacts in EEG signals.

## Features Extraction

- Extracted persistence diagrams from PCA-reduced EEG data using the `ripser` library.
- Converted persistence diagrams into persistence images using the `PersistenceImager` tool.
- Used persistence images as input features for classification.

## Models

- **Random Forest Classifier**
- **K-Nearest Neighbors Classifier**

Both models were tuned via grid search and evaluated on multiple noise scenarios.

## Noise Types Tested

- Noise-Free EEG signals
- Gaussian Noise
- Spike Noise
- Eye-Blink Artifacts

## Results Summary

| Noise Type     | Random Forest Accuracy | KNN Accuracy |
|----------------|------------------------|--------------|
| Noise-Free     | 96%                    | 96%          |
| Gaussian Noise | 96%                    | 96%          |
| Spike Noise    | 94%                    | 94%          |
| Eye-Blink      | 95%                    | 95%          |

## Usage

1. Prepare your EEG dataset with appropriate identifiers.
2. Run the feature extraction pipeline to generate persistence images.
3. Train and tune classifiers using the provided scripts.
4. Evaluate model performance on test data.


