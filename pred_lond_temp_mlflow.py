# Run this cell to import the modules you require

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



# 1. Load the dataset
def load_data(file_path="london_weather.csv"):
    df = pd.read_csv(file_path)
    print("--- Data Loaded Successfully ---")
    return df

df_raw = load_data()

# 2. Inspect the data 
print("--- Data Info ---")
weather.info()

# .describe() provides summary statistics
print("\n--- Summary Statistics ---")
print(weather.describe())

# 3. Data Cleaning & Preparation
# Convert the 'date' column from YYYYMMDD (int) to datetime objects
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')

# Identify missing values before modeling
print("\n--- Missing Values Count ---")
print(weather.isnull().sum())

# 4. Exploratory Visualization
plt.figure(figsize=(12, 6))
sns.lineplot(data=weather, x='date', y='mean_temp', color='tab:red', linewidth=0.5)
plt.title('Daily Mean Temperature - London Dataset')
plt.xlabel('Year')
plt.ylabel('Mean Temp (°C)')
plt.grid(True, alpha=0.3)
plt.show()


# Configure MLflow
mlflow.set_experiment("London_Temperature_Refined_v3")

# 5. Data Cleaning

def clean_data(df):
    # 5.1 Determine names, types, and nulls
    print("\nColumn Metadata:")
    print(df.info())
    
    # 5.2 Working with date column (ISO format)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # 5.3 Extracting more date information (Temporal Features)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Handle the target variable: Drop NaNs in target to avoid model bias
    df = df.dropna(subset=['mean_temp'])
    return df

df_clean = clean_data(df_raw)

# 6. EDA

def perform_eda(df):
    plt.figure(figsize=(12, 5))
    
    # Visualizing Temperature Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df['mean_temp'], kde=True, color='royalblue')
    plt.title("Mean Temperature Distribution")
    
    # Visualizing Temperature over time
    plt.subplot(1, 2, 2)
    plt.plot(df['date'], df['mean_temp'], alpha=0.5, color='orange')
    plt.title("Historical Temperature Series")
    plt.show()

perform_eda(df_clean)

# 7. Feature Selection
# Strategy: Scientific selection based on meteorological drivers + Autoregressive Lags
df_clean['lag_1'] = df_clean['mean_temp'].shift(1) # Yesterday's temp
df_clean = df_clean.dropna() # Drop the first row created by lag

features = ['cloud_cover', 'sunshine', 'global_radiation', 'precipitation', 
            'pressure', 'month', 'year', 'lag_1']
target = 'mean_temp'

X = df_clean[features]
y = df_clean[target]

# 8. Preprocess Data (Pipeline)
# Scientific Split: TimeSeriesSplit avoids "Future Leakage"
# We reserve the last 20% for the final blind test.
split_idx = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

def get_pipeline(regressor):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Step 5.1: Impute
        ('scaler', StandardScaler()),                 # Step 5.2: Normalize
        ('regressor', regressor)
    ])

# 9. Training, Hyperparameters & MlFlow
# Building a search space for multiple models
model_search_space = [
    {
        'name': 'LinearRegression',
        'model': LinearRegression(),
        'params': {}
    },
    {
        'name': 'DecisionTree',
        'model': DecisionTreeRegressor(),
        'params': {'regressor__max_depth': [5, 10]}
    },
    {
        'name': 'RandomForest',
        'model': RandomForestRegressor(),
        'params': {'regressor__n_estimators': [50, 100], 'regressor__max_depth': [10]}
    }
]

# 10. Nested CV Logic using TimeSeriesSplit (The gold standard for climate data)
tscv = TimeSeriesSplit(n_splits=3)

for m in model_search_space:
    with mlflow.start_run(run_name=m['name']):
        pipeline = get_pipeline(m['model'])
        
        # GridSearch within the training split
        grid = GridSearchCV(pipeline, m['params'], cv=tscv, scoring='neg_root_mean_squared_error')
        grid.fit(X_train, y_train)
        
        # Best model evaluation
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Logging to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Logged {m['name']} - RMSE: {rmse:.4f}")


# 11. Searching Logged Results
# This variable fulfills requirement #7
experiment_results = mlflow.search_runs(experiment_names=["London_Temperature_Refined_v3"])

# Output the findings
print("\n--- Final Experiment Results (MLflow) ---")
print(experiment_results[['run_id', 'params.regressor__n_estimators', 'metrics.rmse']].sort_values(by='metrics.rmse'))



