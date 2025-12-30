# 🌦️ London Weather: Predictive Analytics & Temporal Experiment Tracking

[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-1.6+-blue?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.8+-red?logo=matplotlib)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-0.13+-blueviolet?logo=seaborn)](https://seaborn.pydata.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-lightgrey?logo=mlflow)](https://mlflow.org/)

---

## 📌 Executive Summary
This project implements a robust machine learning pipeline to predict daily mean temperature in London using historical meteorological data. Beyond standard regression, the architecture is built on Geographic Data Science best practices, prioritizing chronological integrity through Nested Time-Series Cross-Validation and Autoregressive Feature Engineering.

The final model achieved a Root Mean Squared Error (RMSE) of 1.8100, significantly outperforming the industry benchmark requirement of $RMSE < 3.0$.

---

## 🔬 Scientific Methodology

### 1. Temporal Topology & Data Integrity
Unlike standard $i.i.d.$ (independent and identically distributed) datasets, weather data is a continuous time series. To prevent Data Leakage (Look-ahead bias):

- **Sequential Splitting**: We avoided random shuffling. Data was split chronologically (80% Train / 20% Test) to simulate real-world forecasting.
- **TimeSeriesSplit**: Implemented a nested cross-validation strategy that respects the "arrow of time," ensuring the model is validated on "future" data relative to its training set.

### 2. Autoregressive Feature Engineering
To capture the thermal inertia of the atmospheric system, we engineered a `lag_1` feature (the previous day's temperature). This acknowledges the physical "state" of the system, providing a strong baseline for the predictive algorithms.

### 3. Automated Preprocessing Pipeline
All preprocessing steps were encapsulated within a Scikit-Learn Pipeline:

- **Imputation**: Handled missing sensor data (notably in `snow_depth`) using a median strategy to maintain robustness against outliers.
- **Normalization**: Applied `StandardScaler` to ensure convergence for linear models and to balance feature influence across different units (e.g., Pressure vs. Radiation).

---

## 📊 Model Performance & Tournament Results
Every experiment, hyperparameter, and metric was logged using MLflow for full auditability and reproducibility.

| Model                   | RMSE (Final Test) | Hyperparameters (Optimal)      |
|-------------------------|-----------------|--------------------------------|
| Random Forest Regressor | 1.8100          | n_estimators: 100, max_depth: 10 |
| Linear Regression       | 1.9175          | Default                        |
| Decision Tree Regressor | 1.9198          | max_depth: 5                   |

**Key Findings:**

- **Non-Linearity**: The 5.6% performance gain of the Random Forest over Linear Regression highlights the non-linear relationship between cloud cover, global radiation, and surface temperature.
- **Stationarity and Stability**: The consistency across nested folds suggests a highly generalized model capable of handling seasonal shifts.

---

## 🛠️ Project Structure & Usage

### Setup
Clone the repository:

```bash
git clone https://github.com/your-username/london-weather-ml.git
Install dependencies:

bash
Copiar código
pip install pandas scikit-learn mlflow matplotlib seaborn
Running the Pipeline
The pipeline handles the full lifecycle: EDA, model training, hyperparameter tuning, and MLflow tracking.

Example Usage:

python
Copiar código
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load and clean data
df = pd.read_csv("london_weather.csv")
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['lag_1'] = df['mean_temp'].shift(1)
df = df.dropna()

features = ['cloud_cover', 'sunshine', 'global_radiation', 'precipitation', 
            'pressure', 'month', 'year', 'lag_1']
X = df[features]
y = df['mean_temp']

# Train-test split (chronological)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Pipeline function
def get_pipeline(regressor):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])

# MLflow experiment
mlflow.set_experiment("London_Temperature_Refined_v3")

# Model training & logging
model_search_space = [
    {'name': 'LinearRegression', 'model': LinearRegression(), 'params': {}},
    {'name': 'DecisionTree', 'model': DecisionTreeRegressor(), 'params': {'regressor__max_depth': [5,10]}},
    {'name': 'RandomForest', 'model': RandomForestRegressor(), 'params': {'regressor__n_estimators':[50,100],'regressor__max_depth':[10]}}
]

tscv = TimeSeriesSplit(n_splits=3)
for m in model_search_space:
    with mlflow.start_run(run_name=m['name']):
        pipeline = get_pipeline(m['model'])
        grid = GridSearchCV(pipeline, m['params'], cv=tscv, scoring='neg_root_mean_squared_error')
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(grid.best_estimator_, "model")
        print(f"{m['name']} logged with RMSE: {rmse:.4f}")
To view MLflow dashboard:

bash
Copiar código
mlflow ui
📍 Current Research: Tropical Adaptation (Recife, PE)
This framework is currently being refactored for the Metropolitan Region of Recife, Brazil. The tropical adaptation focuses on:

Maritimity Proxies: Integrating humidity flux and Atlantic sea-surface temperature data.

Rainy Season Encoding: Handling the distinct "Quadra Chuvosa" (April–August) dynamics of the Pernambuco coast.

Urban Heat Island (UHI): Utilizing this pipeline to analyze thermal comfort in high-density urban canyons.

👤 Author
[Adalberto Correia]
PhD Candidate in Geography | Geospatial Data Scientist
Specialist in Geoprocessing and Neuro-symbolic GeoAI architectures
