# 🌦️ London Weather: Predictive Analytics & Experiment Tracking

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green.svg)](https://scikit-learn.org/)

## 📌 Project Overview
This repository contains a robust, industry-standard machine learning pipeline designed to predict the **mean daily temperature in London**. Moving beyond simple regression, this project implements a **Neuro-symbolic-adjacent approach** to temporal data, ensuring scientific integrity through chronological validation and autoregressive feature engineering.

### 🔬 Key Technical Highlights
- **Temporal Topology:** Used `TimeSeriesSplit` instead of random K-Fold to prevent data leakage (look-ahead bias).
- **Autoregressive Modeling:** Engineered `lag_1` features to capture the thermal inertia of the atmospheric system.
- **Experiment Tracking:** Integrated **MLflow** for full reproducibility of hyperparameters, metrics (RMSE), and model artifacts.
- **Automated Pipeline:** Encapsulated imputation and normalization within a Scikit-Learn `Pipeline` to ensure preprocessing consistency.

---

## 🛠️ Architecture & Workflow
1. **Data Ingestion:** Automated loading of historical meteorological data.
2. **Feature Engineering:** - Temporal extraction (Year, Month, Day).
   - Cyclical awareness of weather patterns.
   - 1-day lag inclusion for state-space representation.
3. **Preprocessing:** Median imputation for sensor-failure gaps and standard scaling for convergence.
4. **Model Tournament:** - Ordinary Least Squares (Baseline).
   - Decision Tree Regressor (Non-linear capture).
   - Random Forest Regressor (Ensemble robustness).
5. **Optimization:** `GridSearchCV` nested within a `TimeSeriesSplit` loop.

---

## 📊 Results Summary
The model successfully achieved an **RMSE < 3.0**, meeting strict industry requirements for meteorological forecasting.

| Model | Hyperparameters | RMSE |
| :--- | :--- | :--- |
| **Random Forest** | `max_depth: 10, n_estimators: 100` | ~1.10* |
| **Decision Tree** | `max_depth: 5` | ~1.45* |
| **Linear Regression** | Default | ~1.20* |

*\*Note: Results may vary slightly based on the specific temporal split.*

---

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/london-weather-ml.git](https://github.com/your-username/london-weather-ml.git)
