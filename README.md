<div align="center">

# 📉 Customer Churn Prediction

**ML pipeline** for predicting customer churn with classification models and business metrics

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Tests-7%20passed-success?style=flat-square)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

A **customer churn prediction pipeline** using logistic regression and random forest classifiers. Generates realistic synthetic customer data with feature engineering, model training with stratified train/test splits, and comprehensive evaluation (accuracy, precision, recall, F1, ROC-AUC).

## Features

- 📊 **Synthetic Data Generation** — Realistic customer features (tenure, charges, contracts)
- 🔧 **Preprocessing Pipeline** — Label encoding, standard scaling, feature selection
- 🏆 **Dual Model Training** — Logistic Regression + Random Forest with comparison
- 📈 **Full Evaluation Suite** — Accuracy, Precision, Recall, F1, ROC-AUC
- 🚀 **REST API** — Train and predict endpoints
- 📋 **Interactive Dashboard** — Streamlit UI for model comparison

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
