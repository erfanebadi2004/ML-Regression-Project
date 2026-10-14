# Store Sales Time Series Forecasting MLOPS Project

**Data Source :** Kaggel

## Introduction

This project implements an end-to-end MLOps pipeline for forecasting store product sales using time series data it features
automated model training with lightgbm mlflow expriment traking robust input validation via pydantic high performance API
serving using fastapi and real time data drift monitoring using evidently

## Tools
* **Modeling :** "OneHotEncoder", "StandardScaler", "SimpleImputer", "LightGBM_Regressor"
* **Versioning :** "MLFlow" (Experiment traking & Regitry Model)
* **API Serving :** "FastAPI"
* **Validation :** "Pydantic"
* **Monitoring :** "Evidently AI"
* **Testing :** "pytest"

## Core Components

1. **Pipeline**
- uses column transformer to handel numerical features (SimpleImputer, StandardScaler) and Categorical features (SimpleImputer, OneHotEncoder)
- integrates LightGBMRegressor for time series forecasting

2. **Model Training**
- load data set from csv.
- performs temporal split dased on data 2017-07-15.
- Evaluates the model using R-squared (𝑅2), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- Logs parameters, metrics, and models to MLflow (http://127.0.0.1:5000, Experiment: StoreSalesTimeSeries).
- Saves model artifacts locally in ./LgbmRegressor_folder.

3. **Monitoring**
- Utilizes Evidently (DataDriftPreset) to compare reference data against incoming request data in the background.
- Emits warnings if the feature drift share exceeds 50%.

4. **Input Validation**
- Leverages Pydantic models (FeaturesInput, BatchPredictionRequest) to ensure type safety and structure integrity before inference.

5. **API Serving**
- built with fastapi
- Loads the model via mlflow.pyfunc and pre-saved feature names (features_name.pkl) at startup.
- Implements asynchronous background tasks (BackgroundTasks) to check for data drift without blocking inference.
- 
6. **Deployment**
- Containerizing (Dockerizing) the application for model deployment on servers.

## Project Structure
1. **api_model.py :**            # FastAPI implementation & inference engine
2. **model_training.py :**       # MLflow-integrated training script
3. **pipeline_lgbm.py :**        # Preprocessing & Pipeline definition
4. **validation.py :**           # Pydantic schemas for input validation
5. **data_monitoring.py :**      # Evidently-based Drift Detection
6. **test_api.py :**             # Integration tests
7. **requirements.txt :**        # Project dependencies
8. **Files/ :**                  # Artifacts, Reference data, and Feature names