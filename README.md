# Customer Churn Prediction — End-to-End ML Pipeline

This project is **created for demonstrating an end-to-end machine learning pipeline system** for **customer churn prediction**. I built it with a focus on reproducibility, clean architecture, and explainability.

Instead of treating this as a single notebook experiment, my goal was to design the project simulating how real ML systems are built: configurable training, reusable preprocessing pipelines, proper evaluation,and a deployable inference API.

## Explainability of the entire project

A detailed explainability analysis is available here:
➡️ **[View Explainability Analysis](docs/explainability.md)**, includes SHAP summary plot & Feature importance analysis.

## What this project does

- Trains and evaluates churn prediction models using a real-world telecom dataset
- Progresses from a baseline model to a **tuned XGBoost classifier**
- Uses **cross-validation** and **threshold tuning** instead of assuming a default 0.5 cutoff
- Explains model behavior using **feature importance** and **SHAP**
- Exposes the trained model via a **FastAPI inference service**

## Problem overview

The challenge is not only predicting churn accurately, but also:
- avoiding data leakage
- choosing appropriate evaluation metrics
- explaining predictions in a way that makes sense for customer retention and avoiding business losses.

This project addresses all three.

## Modeling approach

### Baseline
- Logistic Regression (used as a strong, interpretable baseline)

### Advanced models
- Random Forest
- Gradient Boosting
- **XGBoost (final model)**

### Final model
- **XGBoost**, lightly tuned using `RandomizedSearchCV`
- Cross-validation with `average_precision` (PR-AUC) scoring
- Final decision threshold chosen by **maximizing F1**, **not hardcoded**

## Results (tuned XGBoost)

- **Best F1 (threshold-tuned):** ~0.64  
- **Best threshold:** ~0.35  
- **PR-AUC:** ~0.67  

## Inference API

- The trained model is routed via a **FastAPI** service.
- For convinience, I have updated the URL to redirect to `/docs` (interactive swagger UI) for anyone trying out the model in `/predict`

## Docker support

A Dockerfile is included so the service can be run consistently and so that it run everywhere..

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api

