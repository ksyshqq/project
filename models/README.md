

# 🤖 Models Directory

[![Models](https://img.shields.io/badge/Trained%20Models-3-green.svg)]()
[![Best Model](https://img.shields.io/badge/Best-XGBoost-blue.svg)]()
[![Format](https://img.shields.io/badge/Format-.pkl-orange.svg)]()

> This directory contains all trained machine learning models used for cardiovascular disease prediction.

---

## 📦 Trained Models

After running the project, you will find:

* **`logistic_regression_model.pkl`** — Logistic Regression classifier
* **`random_forest_model.pkl`** — Random Forest classifier
* **`xgboost_model.pkl`** — XGBoost classifier *(best performing model)*

---

## 📊 Model Performance Comparison

| Model               | ROC-AUC   | Accuracy  |
| ------------------- | --------- | --------- |
| Logistic Regression | 0.791     | 72.7%     |
| Random Forest       | 0.798     | 73.2%     |
| **XGBoost**         | **0.799** | **73.2%** |

---

## 📝 Notes

* `.pkl` model files are **not stored** in the Git repository (file size).
* All models can be reproduced by running the training pipeline.
* XGBoost was selected as the **production-ready** model due to the highest ROC-AUC score.


✔️ README для **src/models** с архитектурой пайплайна
✔️ нарисовать схему ML-процесса (EDA → preprocessing → training → evaluation)
