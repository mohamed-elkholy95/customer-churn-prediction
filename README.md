<div align="center">

# 📉 Customer Churn Prediction

**End-to-end ML pipeline** for predicting customer churn with multiple classifiers, interactive dashboard, and REST API

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Tests-72%20passed-success?style=flat-square)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

</div>

## Overview

A production-ready customer churn prediction system that demonstrates the full ML lifecycle: data generation, preprocessing, model training, evaluation, feature analysis, and deployment via REST API and interactive dashboard.

Three classification models are compared:
- **Logistic Regression** — fast, interpretable linear baseline
- **Random Forest** — robust bagging ensemble with feature importance
- **Gradient Boosting** — sequential boosting for maximum accuracy

## Features

- 📊 **Synthetic Data Generation** — Realistic customer features with configurable size and reproducible randomness
- 🔧 **Preprocessing Pipeline** — Label encoding + StandardScaler with input validation
- 🏆 **Triple Model Comparison** — Logistic Regression, Random Forest, Gradient Boosting
- 📈 **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC
- 🎯 **Confusion Matrix Analysis** — Visual breakdown of TN/FP/FN/TP with business impact simulation
- 🔍 **Feature Importance** — Ranked feature scores across all three model types
- 🔄 **Cross-Validation** — Stratified k-fold CV for robust performance estimates
- 👤 **Single Customer Scoring** — Real-time churn risk prediction with risk levels (LOW/MEDIUM/HIGH)
- 💾 **Model Persistence** — Save/load trained models with joblib for deployment
- 🚀 **REST API** — 6 FastAPI endpoints with validation and OpenAPI docs
- 📋 **Interactive Dashboard** — 4-page Streamlit app with educational walkthroughs
- ✅ **72 Tests** — Comprehensive pytest suite covering models, API, and edge cases

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Launch dashboard
streamlit run streamlit_app/app.py

# Start API server
uvicorn src.api.main:app --reload --port 8011
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check with version |
| `POST` | `/predict` | Train all models, return evaluation metrics |
| `POST` | `/predict/customer` | Score a single customer's churn risk |
| `POST` | `/feature-importance` | Compute ranked feature importance |
| `POST` | `/cross-validate` | Run stratified k-fold cross-validation |
| `POST` | `/confusion-matrix` | Get confusion matrices for all models |

### Example: Score a Customer

```bash
curl -X POST http://localhost:8011/predict/customer \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25, "tenure": 3, "monthly_charges": 95,
    "total_charges": 285, "contract_type": "month",
    "internet_service": "Fiber", "payment_method": "electronic"
  }'
```

Response:
```json
{
  "customer": { "age": 25, "tenure": 3, "monthly_charges": 95, ... },
  "churn_probability": 0.9527,
  "risk_level": "HIGH",
  "risk_emoji": "🔴",
  "model_used": "gradient_boosting"
}
```

## Dashboard Pages

1. **📊 Overview** — What is churn, ML pipeline walkthrough, model explanations, metric definitions
2. **📈 Train & Evaluate** — Interactive data generation, preprocessing visualization, model training and comparison
3. **🔍 Feature Importance** — Ranked feature scores with business interpretation for each model
4. **🎯 Confusion Matrix** — Visual confusion matrices with business impact cost simulator

## Project Structure

```
├── src/
│   ├── churn_model.py      # Core ML pipeline (data gen, preprocessing, training, evaluation)
│   ├── config.py            # Centralized configuration and constants
│   └── api/
│       └── main.py          # FastAPI REST API (6 endpoints)
├── tests/
│   ├── test_churn.py        # 57 model tests (generation, preprocessing, evaluation, persistence)
│   └── test_api.py          # 15 API endpoint tests
├── streamlit_app/
│   ├── app.py               # Dashboard entry point with navigation
│   └── pages/               # 4 interactive pages
├── examples/
│   ├── quickstart.py        # Getting started script
│   └── demo.py              # Feature demonstration
├── docs/
│   ├── ARCHITECTURE.md      # System design and data flow
│   ├── DEVELOPMENT.md       # Setup, testing, and code style
│   ├── CONTRIBUTING.md      # Contribution guidelines
│   ├── GLOSSARY.md          # ML terminology reference (25+ terms)
│   └── PROJECT_PLAN.md      # Development roadmap
├── requirements.txt
└── README.md
```

## Key Concepts

This project is designed as an educational portfolio piece. Key ML concepts demonstrated:

- **Stratified splitting** — preserves class balance in train/test/CV folds
- **Feature importance** — different computation methods for linear vs tree models
- **Confusion matrix analysis** — business cost tradeoff between false positives and false negatives
- **Cross-validation** — robust performance estimation vs single train/test split
- **Model persistence** — saving and loading trained models for production deployment

See [docs/GLOSSARY.md](docs/GLOSSARY.md) for a complete terminology reference.

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
