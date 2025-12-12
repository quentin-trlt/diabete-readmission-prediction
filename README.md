# Diabetes Hospital Readmission Prediction - Premier Rendu Intermédiaire

## Problem Statement
Can we predict 30-day hospital readmission risk for diabetic patients based on their clinical history and medication management?

## Project Structure

```
.
├── data/
│   ├── diabetic_data_preprocessed.csv
│   └── diabetic_data.csv          
├── ml_models/                      
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── xgboost.py                 
├── data_analysis.ipynb  
├── preprocessing.ipynb 
├── model_training.ipynb 
├── prediction.ipynb 
├── requirements.txt 
└── README.md                      
```


## Notebooks

### 1. data_analysis.ipynb
Exploratory data analysis of the diabetes readmission dataset. Examines distributions, missing values, feature correlations, and class imbalance patterns.

### 2. preprocessing.ipynb
Data cleaning and feature engineering pipeline. Handles missing values, encodes categorical variables, creates derived features, and prepares the final dataset for modeling.

### 3. model_training.ipynb
Trains and optimizes three models (Logistic Regression, Random Forest, XGBoost) using Bayesian hyperparameter search and SMOTE resampling. Evaluates performance using F2-score, recall, and precision metrics. Includes cost-benefit analysis.

### 4. prediction.ipynb
Demonstrates the prediction workflow using trained models. Implements three-level risk stratification (Low/Medium/High) and provides clinical interpretation with feature importance analysis.

## ML Models Package

The `ml_models/` package contains reusable model implementations with consistent APIs:
- **logistic_regression.py**: Baseline linear model with L2 regularization
- **random_forest.py**: Ensemble tree-based model with balanced class weights
- **xgboost.py**: Gradient boosting implementation optimized for imbalanced classification

Each module provides a standardized interface for training, prediction, and evaluation.

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
jupyter>=1.0.0
notebook>=6.4.0
plotly
xgboost
scikit-optimize>=0.9.0
```

## References

Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. BioMed Research International, Article ID 781670.

## Authors

Machine Learning Project - Healthcare Analytics