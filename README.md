# Credit Card Fraud Detection Using Machine Learning

## Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning classification algorithms.

It explores the complete ML workflow, including data preprocessing, exploratory data analysis, handling imbalanced datasets, model training, performance evaluation, and model explainability using SHAP.

## Dataset

**Source:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains transactions made by European cardholders in September 2013.

- Total transactions: 284,807
- Fraudulent transactions: 492
- Legitimate transactions: 284,315
- Target variable: `Class`
  - `0` – Legitimate transaction
  - `1` – Fraudulent transaction

The dataset is highly imbalanced, making fraud detection a challenging classification problem.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Imbalanced Data Handling:** SMOTE
- **Gradient Boosting:** XGBoost, LightGBM
- **Model Explainability:** SHAP
- **Environment:** Kaggle Notebook (NVIDIA T4 GPU)

## Machine Learning Models

Four classification algorithms were implemented:

1. Logistic Regression – Baseline Model
2. Random Forest
3. XGBoost
4. LightGBM

## Project Workflow

1. Data Loading and Exploratory Data Analysis (EDA)
2. Feature Engineering and Preprocessing
3. Train-Test Split with Stratified Sampling
4. Feature Scaling
5. Handling Class Imbalance using SMOTE
6. Training Multiple Classification Models
7. Model Performance Comparison
8. Model Evaluation using Confusion Matrix, ROC-AUC, Precision, Recall, and F1-Score
9. ROC and Precision-Recall Curve Visualization
10. SHAP-Based Feature Importance and Model Explainability
11. Saving Trained Models using Pickle (`.pkl`)

## Model Explainability – SHAP

SHAP (SHapley Additive exPlanations) was used to understand the contribution of individual features to the XGBoost model's predictions.

The analysis includes:

- SHAP Feature Importance (Mean Absolute SHAP Values)
- SHAP Beeswarm Plot
- Feature Impact Direction

Top 5 features identified by SHAP:

1. V14
2. V4
3. V1
4. V8
5. V12

## Model Evaluation

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Precision-Recall AUC

## Project Structure

```text
credit-card-fraud-detection/
│
├── credit-card-fraud-detection-model.ipynb
├── README.md
│
└── models/
    ├── LogisticRegression.pkl
    ├── RandomForest.pkl
    ├── XGBoost.pkl
    └── LightGBM.pkl
```

The trained model files are optional and are not required to execute the notebook.

## How to Run

1. Clone this repository.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3. Open the notebook using Kaggle or Jupyter Notebook.
4. Upload the dataset and configure the dataset path.
5. Execute the notebook cells sequentially.

## Future Improvements

- Hyperparameter Optimization
- Cross-Validation
- Improved Classification Threshold Optimization
- Model Deployment using FastAPI
- Real-Time Fraud Detection
