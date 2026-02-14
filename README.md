# Neural Network Regression Model – Tabular Data

A deep learning–based regression model built using TensorFlow/Keras to predict continuous target values from structured tabular data.

The project demonstrates model design, regularization, optimization techniques, and performance evaluation using advanced regression metrics.

# Project Overview

This project focuses on:

Building a deep neural network for regression

Preventing overfitting using Batch Normalization + Dropout

Improving generalization with learning rate scheduling

Evaluating performance using R², MAE, and MSE

Comparing validation vs test behavior for robustness

The final model achieves:

Test MSE: 0.316  
Test MAE: 0.375  
R² Score: 0.771

# Model Architecture

Dense(128, activation='relu')

BatchNormalization()
Dropout(0.3)

Dense(64, activation='relu')

BatchNormalization()

Dropout(0.3)

Dense(1)

# Why this architecture?

128 → 64 layers: Increased model capacity to reduce underfitting

Batch Normalization: Stabilizes training and improves convergence

Dropout (0.3): Prevents overfitting

ReduceLROnPlateau: Adaptive learning rate control

📊 Performance Metrics

| Metric | Score |
| ------ | ----- |
| MSE    | 0.316 |
| MAE    | 0.375 |
| R²     | 0.771 |

#  Interpretation

R² = 0.771 → Model explains 77% of variance

Test loss ≈ Validation loss → Strong generalization

No major overfitting observed

# Training Strategy

Optimizer: Adam

Loss Function: Mean Squared Error

Learning Rate Scheduler: ReduceLROnPlateau

Early stopping (if applied)

Standardized input features


# Experimentation & Improvements

| Version               | R² Score |
| --------------------- | -------- |
| Initial Model         | 0.736    |
| Improved Architecture | 0.771    |


3 Key improvement:

Increased hidden units

Better representation capacity

Reduced underfitting

# Tech Stack

Python

NumPy

Pandas

Matplotlib

Scikit-learn

TensorFlow / Keras

# Repository Structure

NeuroPrice/

├── NeuroPrice_Deep_Learning_Housing_Price_Predictor.ipynb

├── california_housing_regressor.keras

├── requirements.txt

└── README.md


# Key Learning Outcomes

Deep learning for tabular data

Regularization techniques

Hyperparameter tuning

Learning rate scheduling

Model evaluation using regression metrics

Generalization analysis


# 🔮 Future Improvements

Compare with Random Forest & XGBoost

Hyperparameter tuning using Keras Tuner

Add cross-validation

Deploy with Streamlit

Add SHAP for model explainability



