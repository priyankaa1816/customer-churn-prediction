# Customer Churn Prediction using XGBoost

This project focuses on predicting **customer churn** using machine learning techniques on structured tabular data.  
The goal is to identify customers who are likely to exit a service, enabling businesses to take **proactive retention actions**.

The solution is built using a **clean, end-to-end ML pipeline** with proper preprocessing, class imbalance handling, and robust evaluation.

---

## Problem Overview

Customer churn occurs when users stop using a company’s products or services.  
High churn rates negatively impact revenue and long-term growth, making churn prediction a critical business problem.

This project formulates churn prediction as a **binary classification problem**, where the target variable indicates whether a customer has exited (`Exited = 1`) or not (`Exited = 0`).

---

## Dataset

The dataset is sourced from a Kaggle competition and is **synthetically generated** based on a real-world bank customer churn dataset.

### Files:
- `train.csv` – Training data with the target variable `Exited`
- `test.csv` – Test data without the target
- `sample_submission.csv` – Submission format reference

---

## Methodology

### 🔹 Preprocessing
- Numerical features are **standardized** using `StandardScaler`
- Categorical features are **one-hot encoded**
- **SMOTE** is applied within the pipeline to handle class imbalance safely

### 🔹 Model
- **XGBoost (Gradient Boosted Trees)** is used due to its strong performance on tabular data
- Hyperparameters are manually tuned to balance performance and stability

### 🔹 Evaluation Strategy
- **Stratified 5-Fold Cross-Validation**
- **ROC-AUC** is used as the primary evaluation metric due to class imbalance

---

## Results

- **Mean Cross-Validation ROC-AUC:** ~0.93  
- **Validation ROC-AUC (Hold-out set):** ~0.94  

These results indicate strong discriminative performance and good generalization.

---

## Key Learnings

- Importance of **pipeline-based design** to avoid data leakage  
- Proper handling of **imbalanced datasets** using SMOTE  
- ROC-AUC is a more reliable metric than accuracy for churn prediction  
- XGBoost effectively captures non-linear feature interactions in tabular data  

---

## Future Improvements

- Perform systematic hyperparameter tuning in a separate experimentation notebook  
- Explore feature selection to simplify the model  
- Compare performance with LightGBM or CatBoost  
- Integrate the model into a real-time churn monitoring system  

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn  
- XGBoost  
- Matplotlib, Seaborn  

---


