# Breast Cancer Prediction using KNN

## Overview
This project leverages the K-Nearest Neighbors (KNN) algorithm to predict the likelihood of breast cancer based on data from the UCI Machine Learning Repository. The dataset includes features computed from breast cancer biopsies, such as mean, standard error, and the "worst" or largest values for various cell features.

## Project Structure
1. **Data Preprocessing**:
   - Cleaned the dataset by dropping unwanted columns such as `id` and unnamed columns.
   - Converted categorical diagnosis labels (`M` for malignant and `B` for benign) into numerical values for modeling.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized relationships between features like `radius_mean`, `texture_mean`, `smoothness_mean`, and `compactness_mean` using scatter plots.

3. **Model Training**:
   - Used the K-Nearest Neighbors algorithm with `n_neighbors=7` after testing multiple values for optimal performance.
   - Trained the model on 70% of the dataset and tested on the remaining 30%.

4. **Evaluation**:
   - Achieved an accuracy of **96.49%** on the test set.
   - The model’s performance was evaluated using metrics such as Root Mean Squared Error (RMSE) and R² score.

## Key Results
- **Accuracy**: 96.49%
- **RMSE**: 0.1873
- **R² Score**: 0.8492

## Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset
The dataset is taken from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
