<div align="center">

# 📉 Customer Churn Prediction

**End-to-end ML pipeline** for predicting customer churn with multiple classifiers, interactive dashboard, and REST API

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Tests-101%20passed-success?style=flat-square)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

</div>

## Overview

A production-ready customer churn prediction system that demonstrates the full ML lifecycle: data generation, preprocessing, model training, evaluation, feature analysis, threshold optimization, and deployment via REST API and interactive dashboard.

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
- ⚙️ **Threshold Optimization** — Find the optimal classification threshold for any metric
- 📈 **Learning Curves** — Diagnose underfitting/overfitting with training size analysis
- 🏅 **Automated Model Recommendation** — Compare all models and get a deployment recommendation
- 👤 **Single Customer Scoring** — Real-time churn risk prediction with risk levels (LOW/MEDIUM/HIGH)
- 💾 **Model Persistence** — Save/load trained models with joblib for deployment
- 🌐 **Environment Configuration** — All settings overridable via environment variables
- 🚀 **REST API** — 9 FastAPI endpoints with validation, logging middleware, and OpenAPI docs
- 📋 **Interactive Dashboard** — 5-page Streamlit app with educational walkthroughs
- ✅ **101 Tests** — Comprehensive pytest suite covering models, API, and edge cases

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt

# Run examples
python examples/quickstart.py    # Core pipeline in 30 lines
python examples/demo.py          # Full feature walkthrough

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
| `POST` | `/compare` | Compare all models and get deployment recommendation |
| `POST` | `/optimal-threshold` | Find the best classification threshold for a metric |
| `POST` | `/learning-curve` | Compute learning curve for training size analysis |

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

### Example: Find Optimal Threshold

```bash
curl -X POST http://localhost:8011/optimal-threshold \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gradient_boosting", "metric": "f1"}'
```

Response:
```json
{
  "optimal_threshold": 0.38,
  "best_score": 0.7654,
  "metric": "f1",
  "model_name": "gradient_boosting"
}
```

## Dashboard Pages

1. **📊 Overview** — What is churn, ML pipeline walkthrough, model explanations, metric definitions
2. **📈 Train & Evaluate** — Interactive data generation, preprocessing visualization, model training and comparison
3. **🔍 Feature Importance** — Ranked feature scores with business interpretation for each model
4. **🎯 Confusion Matrix** — Visual confusion matrices with business impact cost simulator
5. **⚙️ Threshold & Learning** — Threshold optimization sweep and learning curve diagnostics

## Environment Configuration

All settings are configurable via environment variables for deployment flexibility:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHURN_RANDOM_SEED` | `42` | Reproducibility seed for all random operations |
| `CHURN_API_HOST` | `0.0.0.0` | Server bind address |
| `CHURN_API_PORT` | `8011` | Server port |
| `CHURN_LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |
| `CHURN_DEFAULT_SAMPLES` | `1000` | Default synthetic dataset size |
| `CHURN_DEFAULT_TEST_SIZE` | `0.2` | Default train/test split ratio |

## Project Structure

```
├── src/
│   ├── churn_model.py      # Core ML pipeline (13 functions)
│   ├── config.py            # Centralized config with env var overrides
│   └── api/
│       └── main.py          # FastAPI REST API (9 endpoints + logging middleware)
├── tests/
│   ├── test_churn.py        # 69 model tests
│   └── test_api.py          # 32 API endpoint tests
├── streamlit_app/
│   ├── app.py               # Dashboard entry point with navigation
│   └── pages/               # 5 interactive pages
├── examples/
│   ├── quickstart.py        # Pipeline demo in 30 lines
│   └── demo.py              # Complete feature walkthrough (8 sections)
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
- **Threshold optimization** — finding the decision boundary that minimizes business costs
- **Learning curves** — diagnosing underfitting vs overfitting with training size analysis
- **Model persistence** — saving and loading trained models for production deployment
- **Environment configuration** — twelve-factor app methodology for portable deployments

See [docs/GLOSSARY.md](docs/GLOSSARY.md) for a complete terminology reference.

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
