# EEG Epileptic Seizure Classification using Topological Data Analysis (TDA)

## Features Extraction

- Extracted persistence diagrams from PCA-reduced EEG data using the `ripser` library.
- Converted persistence diagrams into persistence images using the `PersistenceImager` tool.
- Used persistence images as input features for classification.

## Models

- **CatBoost**

precision:  0.94     recall: 0.73  f1-score: 0.82  

                