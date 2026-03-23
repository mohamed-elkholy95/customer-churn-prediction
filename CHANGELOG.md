# Changelog

All notable changes to this project are documented in this file.

## [2.0.0] - 2026-03-23

### Added
- **Single Customer Prediction** — `predict_single_customer()` scores individual customers with churn probability and risk level (LOW/MEDIUM/HIGH)
- **Confusion Matrix Analysis** — `get_confusion_matrices()` returns TN/FP/FN/TP breakdown for all models
- **Model Persistence** — `save_model()` and `load_model()` for serializing trained models, scalers, and metadata with joblib
- **Data Validation** — `validate_dataframe()` checks for missing columns, empty data, and invalid churn values
- **API Endpoints** — 4 new REST endpoints: `/predict/customer`, `/feature-importance`, `/cross-validate`, `/confusion-matrix`
- **Confusion Matrix Dashboard Page** — Interactive Plotly heatmaps with business impact cost simulator
- **ML Glossary** — 25+ term reference guide in `docs/GLOSSARY.md`
- **72 Tests** — Comprehensive pytest suite covering models, API, edge cases, and persistence
- **Constants** — `SUPPORTED_MODELS` and `EXPECTED_COLUMNS` for consistent model/data validation

### Changed
- Rewrote all docstrings with examples, parameter descriptions, and educational context
- Expanded `/predict` endpoint to accept `test_size` parameter and return `churn_rate`
- Added `/health` endpoint version field for deployment tracking
- Improved request validation with field-level constraints and descriptive error messages
- Bumped API version to 2.0.0

### Documentation
- Complete README rewrite with API reference, curl examples, and project structure
- Added inline educational comments throughout `churn_model.py`
- Expanded `config.py` with setting explanations and `MODELS_DIR` constant

## [1.0.0] - 2026-03-22

### Added
- Core churn prediction pipeline with Logistic Regression, Random Forest, and Gradient Boosting
- Synthetic data generation with logistic function for realistic churn labels
- Preprocessing pipeline with label encoding and standard scaling
- Feature importance and cross-validation support
- 3-page Streamlit dashboard with educational walkthroughs
- FastAPI REST API with health check and prediction endpoints
- Architecture, development, and contributing documentation
