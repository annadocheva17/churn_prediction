# Churn Prediction for Streaming Service

**team:** Anna Docheva & Eleanor Liard
**Context:** Kaggle Competition Python for Datascience Final Project

## Project Overview

This repository contains our machine learning model to predict user churn for a music streaming service. The goal is to identify users who will visit the **'Cancellation Confirmation'** page within a **10-day prediction window** (Nov 20 – Nov 30, 2018), based on their activity logs.

Our solution focuses on robust temporal validation and high-recall optimization.

## Repository Structure

| File | Description |
| :--- | :--- |
| `main.py` | **Entry point.** Orchestrates the data loading, window generation, training, and prediction pipeline|
| `config.py` | Global configuration (Dates, Hyperparameters, Paths) |
| `building_datasets.py` | Logic for creating **sliding windows** for temporal cross-validation |
| `features.py` | Feature engineering logic (Engagement trends, Ratios, Activity flags) |
| `model_xgb.py` | **XGBoost** implementation with **Optuna** hyperparameter tuning and GroupKFold validation. |
| `preprocessing.py` | Data cleaning and XGBoost-compatible column formatting |
| `feature_selection.py` | `SelectKBest` logic to reduce overfitting. |
| `evaluation.py` | Metrics calculation and Threshold Optimization (Recall-focused) |
| `labels.py` | Extraction of churn labels based on 'Cancellation Confirmation' page visits. |
| `eda_churn.ipynb` | Exploratory Data Analysis notebook|
| `auto_ml.py` | AutoML tests to choose the best model |

## Methodology

### Temporal Validation Strategy: To prevent data leakage and simulate real-world performance, we implemented a **Sliding Window** approach:

  * **Observation Window:** 21 Days
  * **Prediction Window:** 10 Days
  * **Step Size:** 7 Days
  * **Selected Features** 30
  * **Leakage Prevention:** We utilize `GroupKFold` and `GroupShuffleSplit` to ensure that all windows belonging to a single user are strictly kept in the same fold (Training or Validation)

### Feature Engineering

We extracted ~40 features focusing on behavioral changes:

  * **Trend Analysis:** We split the 25-day window into **thirds**. We calculate the delta in session counts between the 1st, 2nd, and 3rd periods to detect decline in usage
  * **Interaction Quality:** Ratios such as `Songs per Session` and `Thumbs Down Ratio`
  * **Recency:** `Days since last activity` relative to the prediction cutoff

### Model & Optimization

  * **Algorithm:** XGBoost Classifier.
  * **Tuning:** Hyperparameters (Learning rate, Max depth, Scale pos weight) optimized using **Optuna** (150 trials)
  * **Objective:** We optimized for **AUC** but selected the final decision threshold based on **Recall** (\>0.55) to ensure we capture the minority churn class.

### Hybrid Prediction Strategy

We observed that machine learning models often hallucinate patterns on inactive users (users with 0 events in the observation window)

  * **Active Users:** Predictions via XGBoost Model.
  * **Inactive Users:** Predictions via a deterministic probability rule based on historical inactive churn rates

## Key Results

  * **Validation AUC:** 
  * **Recall at Threshold:** 
