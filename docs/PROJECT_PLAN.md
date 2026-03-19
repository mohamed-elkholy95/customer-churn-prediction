# Project Plan: Customer Churn Prediction

> **Status: COMPLETE**  
> Full implementation at: `/home/mac/projects/customer-churn-prediction/`

This document summarizes the completed project. The actual code, notebooks, and artifacts live in the dedicated project directory.

---

## Problem Statement

Predict whether a telecom customer will churn (cancel service) based on their account history, demographics, and usage patterns. Early identification of at-risk customers enables targeted retention campaigns.

**Business Value:** Acquiring new customers costs 5–7× more than retaining existing ones. A 5% reduction in churn can increase profits by 25–95%.

## Dataset

- **Source:** IBM Telco Customer Churn (Kaggle / ICP4D)
- **Size:** 7,043 customers, 21 features
- **Target:** Binary — `Churn` (Yes/No)
- **Class Balance:** ~26.5% churn rate (moderate imbalance)

### Key Features
- Demographics: gender, senior citizen, partner, dependents
- Account: tenure, contract type, payment method, monthly/total charges
- Services: phone, internet, streaming, online backup, tech support

## Implementation Summary

### Phase 1: EDA & Data Prep ✅
- Univariate and bivariate analysis
- Correlation heatmap, churn rate by segment
- Missing value handling (TotalCharges: 11 nulls filled with 0)
- Label encoding + one-hot encoding for categoricals

### Phase 2: Feature Engineering ✅
- `tenure_bin`: bins (0-12, 12-24, 24-48, 48-60, 60+)
- `monthly_x_tenure`: interaction feature
- `num_services`: count of active add-on services
- Standardization via `StandardScaler`

### Phase 3: Class Imbalance ✅
- SMOTE oversampling on training set only
- Class weight parameter tuning (`class_weight='balanced'`)

### Phase 4: Models ✅
| Model | File |
|---|---|
| Logistic Regression | `src/models/logistic.py` |
| XGBoost | `src/models/xgboost_model.py` |
| Cross-validation | `src/evaluation/cross_val.py` |

### Phase 5: Interpretability ✅
- Global SHAP feature importance
- Individual SHAP waterfall plots
- Partial dependence plots for top features
- All plots in `plots/`

### Phase 6: Streamlit Dashboard ✅
- Customer churn probability input form
- Model comparison bar charts
- Segment analysis (churn rate by contract type, tenure, etc.)
- SHAP explanation for individual predictions

### Phase 7: Testing ✅
- `pytest` suite in `tests/`
- Unit tests for feature engineering
- Integration tests for model pipeline
- Coverage report via `pytest --cov=src`

## File Reference

```
/home/mac/projects/customer-churn-prediction/
├── data/
│   ├── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
├── src/
│   ├── data/loader.py
│   ├── features/engineer.py
│   ├── models/logistic.py, xgboost_model.py
│   └── evaluation/metrics.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── streamlit_app/app.py
└── tests/
```

## Tech Stack

`Python 3.11` `scikit-learn` `XGBoost` `SHAP` `imbalanced-learn` `Streamlit` `Plotly` `pandas` `numpy` `pytest`
