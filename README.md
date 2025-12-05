# California Housing Price Prediction — Machine Learning Project

End-to-end regression project using scikit-learn and XGBoost.  
Includes data preprocessing, feature engineering, model comparison, cross-validation, hyperparameter tuning, and final model export.

## Project Overview

This project builds a machine learning system that predicts median house prices in California using the California Housing dataset.

The notebook demonstrates a complete ML workflow:

- Data loading and cleaning  
- Exploratory analysis  
- Preprocessing pipelines  
- Model comparison  
- Cross-validation  
- Grid Search hyperparameter tuning  
- Final evaluation on a test set  
- Saving the best model for deployment  

The implementation uses professional engineering practices with reusable pipelines and functions.

## Dataset

The dataset comes from Aurélien Géron's GitHub (used in Hands-On Machine Learning).

It contains:

- 20,640 rows  
- 10 features (numeric and categorical)  
- Geographic and demographic indicators  
- Target: median_house_value

Key features:

- longitude, latitude  
- housing_median_age  
- total_rooms, total_bedrooms  
- population, households  
- median_income  
- ocean_proximity (categorical)

## Tech Stack

- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- scikit-learn  
- XGBoost  
- Joblib

## Preprocessing

A ColumnTransformer handles different data types:

### Numerical pipeline:
- Median imputation  
- Standard scaling  

### Categorical pipeline:
- Most frequent imputation  
- One-hot encoding  

All preprocessing is wrapped in a pipeline to prevent data leakage.

## Models Tested

The project compares the following regression models:

### Boosting Methods
- XGBoost Regressor  
- Gradient Boosting Regressor  

### Ensemble Trees
- Random Forest Regressor  
- Extra Trees Regressor  

### Simpler Models
- Decision Tree Regressor  
- SVR (RBF)

All models are evaluated using:

- Cross-validation  
- RMSE  
- MAE  
- R2 score  
- Grid Search hyperparameter optimization

## Model Performance — Cross-Validation (Grid Search Best Scores)

| Model | Best CV R2 | Performance (%) |
|-------|------------|------------------|
| XGBoost | 0.8385 | 83.85 |
| Gradient Boosting | 0.8244 | 82.44 |
| Random Forest | 0.8111 | 81.11 |
| Extra Trees | 0.8031 | 80.31 |
| Decision Tree | 0.7284 | 72.83 |
| SVR (RBF) | 0.2894 | 28.94 |

XGBoost consistently outperforms all other models.

## Final Test Set Evaluation

Using the best model from GridSearchCV on unseen test data:

- Model: XGBoost  
- R2: approximately 0.84  
- RMSE: approximately 49,000  
- MAE: approximately 32,000  
- Overall performance: approximately 84 percent  

This confirms that XGBoost generalizes best among all tested algorithms.

## Saving the Final Model

The final XGBoost pipeline (including preprocessing) is saved as:

best_model_xgboost.joblib

This allows reuse in prediction scripts, APIs, or deployment systems.

## Project Structure

california-housing-ml-project/
│
├── Housing_clean.ipynb
├── best_model_xgboost.joblib
├── README.md
└── requirements.txt

## Key Skills Demonstrated

This project shows practical ML engineering skills:

- Correct data splitting to avoid leakage  
- Clean preprocessing using pipelines  
- Using ColumnTransformer for mixed data types  
- Cross-validation and model selection  
- Hyperparameter tuning with Grid Search  
- Comparing multiple ML models  
- Saving and exporting pipelines  
- Creating a reproducible workflow  

## Future Improvements

- Add model interpretability using SHAP  
- Add geographic visualizations  
- Create a prediction API using Flask or FastAPI  
- Package the project using Docker  
- Add feature importance analysis  

## Author

Created by Nur  
Machine Learning Student and aspiring ML Engineer
