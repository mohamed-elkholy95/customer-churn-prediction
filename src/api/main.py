"""
Churn Prediction REST API
=========================

A FastAPI application exposing the churn prediction pipeline as HTTP endpoints.

Endpoints:
    GET  /health                — Service health check
    POST /predict               — Train models and return evaluation metrics
    POST /predict/customer      — Score a single customer's churn risk
    POST /feature-importance    — Compute feature importance for a given model
    POST /cross-validate        — Run k-fold cross-validation across all models
    POST /confusion-matrix      — Get confusion matrices for all models

All training endpoints generate synthetic data on-the-fly. In production,
you would replace this with a database query or file upload.

CORS is enabled for all origins to support frontend development.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    version="2.0.0",
    description=(
        "ML-powered customer churn prediction API. Train models, evaluate "
        "performance, compute feature importance, and score individual customers."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for model training and evaluation."""
    n_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of synthetic customers to generate for training.",
    )
    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of data held out for testing (0-1).",
    )


class CustomerPredictRequest(BaseModel):
    """Request body for single-customer churn prediction."""
    age: int = Field(ge=18, le=100, description="Customer age")
    tenure: int = Field(ge=0, le=120, description="Months as customer")
    monthly_charges: float = Field(ge=0, description="Monthly bill amount")
    total_charges: float = Field(ge=0, description="Lifetime total charges")
    contract_type: str = Field(
        description="Contract type: 'month', 'year', or 'two_year'"
    )
    internet_service: str = Field(
        description="Internet service: 'DSL', 'Fiber', or 'No'"
    )
    payment_method: str = Field(
        description="Payment method: 'auto', 'check', or 'electronic'"
    )
    model_name: str = Field(
        default="gradient_boosting",
        description="Model to use for prediction.",
    )
    n_training_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of training samples to fit the model on.",
    )


class FeatureImportanceRequest(BaseModel):
    """Request body for feature importance computation."""
    n_samples: int = Field(default=1000, ge=100, le=10000)
    model_name: str = Field(
        default="random_forest",
        description="Model for importance: logistic_regression, random_forest, gradient_boosting",
    )


class CrossValidateRequest(BaseModel):
    """Request body for cross-validation."""
    n_samples: int = Field(default=1000, ge=100, le=10000)
    n_splits: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of CV folds (higher = more robust, slower).",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint. Returns service status and version."""
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/predict")
async def predict(req: PredictRequest):
    """Train all three models and return evaluation metrics.

    Generates synthetic data, trains Logistic Regression, Random Forest, and
    Gradient Boosting, evaluates each on a held-out test set, and returns
    accuracy, precision, recall, F1, and ROC-AUC for each model.
    """
    from src.churn_model import generate_synthetic_churn_data, train_and_evaluate

    df = generate_synthetic_churn_data(n_samples=req.n_samples)
    results = train_and_evaluate(df, test_size=req.test_size)

    return {
        "n_samples": len(df),
        "test_size": req.test_size,
        "churn_rate": round(float(df["churn"].mean()), 4),
        "results": results,
    }


@app.post("/predict/customer")
async def predict_customer(req: CustomerPredictRequest):
    """Score a single customer's churn risk.

    Returns a churn probability (0-1) and risk level (LOW/MEDIUM/HIGH).
    The model is trained on synthetic data before scoring.
    """
    from src.churn_model import generate_synthetic_churn_data, predict_single_customer

    valid_models = ["logistic_regression", "random_forest", "gradient_boosting"]
    if req.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name: {req.model_name}. Choose from: {valid_models}",
        )

    valid_contracts = ["month", "year", "two_year"]
    if req.contract_type not in valid_contracts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid contract_type: {req.contract_type}. Choose from: {valid_contracts}",
        )

    df = generate_synthetic_churn_data(n_samples=req.n_training_samples)
    customer = {
        "age": req.age,
        "tenure": req.tenure,
        "monthly_charges": req.monthly_charges,
        "total_charges": req.total_charges,
        "contract_type": req.contract_type,
        "internet_service": req.internet_service,
        "payment_method": req.payment_method,
    }

    result = predict_single_customer(customer, df, model_name=req.model_name)
    return {"customer": customer, **result}


@app.post("/feature-importance")
async def feature_importance(req: FeatureImportanceRequest):
    """Compute and return feature importance scores.

    Shows which customer attributes most strongly predict churn,
    ranked by normalized importance score (sums to 1.0).
    """
    from src.churn_model import generate_synthetic_churn_data, get_feature_importance

    valid_models = ["logistic_regression", "random_forest", "gradient_boosting"]
    if req.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name: {req.model_name}. Choose from: {valid_models}",
        )

    df = generate_synthetic_churn_data(n_samples=req.n_samples)
    importance = get_feature_importance(df, model_name=req.model_name)

    # Sort by importance descending for easier consumption
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "model": req.model_name,
        "n_samples": req.n_samples,
        "importance": sorted_importance,
    }


@app.post("/cross-validate")
async def cross_validate(req: CrossValidateRequest):
    """Run stratified k-fold cross-validation for all models.

    Returns mean and standard deviation of accuracy and F1 score
    for each model, providing robust performance estimates.
    """
    from src.churn_model import generate_synthetic_churn_data, cross_validate_models

    df = generate_synthetic_churn_data(n_samples=req.n_samples)
    results = cross_validate_models(df, n_splits=req.n_splits)

    return {
        "n_samples": req.n_samples,
        "n_splits": req.n_splits,
        "results": results,
    }


@app.post("/confusion-matrix")
async def confusion_matrices(req: PredictRequest):
    """Get confusion matrices for all three models.

    Returns TN, FP, FN, TP counts for each model, showing the
    distribution of correct and incorrect predictions.
    """
    from src.churn_model import generate_synthetic_churn_data, get_confusion_matrices

    df = generate_synthetic_churn_data(n_samples=req.n_samples)
    results = get_confusion_matrices(df, test_size=req.test_size)

    return {
        "n_samples": req.n_samples,
        "test_size": req.test_size,
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8011)
