import pytest
import pandas as pd
from src.churn_model import generate_synthetic_churn_data, preprocess, train_and_evaluate, get_feature_importance, cross_validate_models

class TestGenerate:
    def test_shape(self):
        df = generate_synthetic_churn_data(100)
        assert df.shape == (100, 9)

    def test_has_churn(self):
        assert "churn" in generate_synthetic_churn_data().columns

    def test_reproducible(self):
        assert generate_synthetic_churn_data(seed=42).equals(generate_synthetic_churn_data(seed=42))

    def test_correct_columns(self):
        df = generate_synthetic_churn_data(50)
        expected = {"customer_id", "age", "tenure", "monthly_charges", "total_charges",
                    "contract_type", "internet_service", "payment_method", "churn"}
        assert set(df.columns) == expected

    def test_churn_values_binary(self):
        df = generate_synthetic_churn_data(500)
        assert set(df["churn"].unique()).issubset({0, 1})


class TestPreprocess:
    def test_shapes(self):
        df = generate_synthetic_churn_data(200)
        X, y, _ = preprocess(df)
        assert X.shape[0] == 200


class TestTrainEvaluate:
    def test_returns_results(self):
        df = generate_synthetic_churn_data(200)
        results = train_and_evaluate(df)
        assert "logistic_regression" in results
        assert "roc_auc" in results["logistic_regression"]

    def test_returns_three_models(self):
        df = generate_synthetic_churn_data(300)
        results = train_and_evaluate(df)
        assert "logistic_regression" in results
        assert "random_forest" in results
        assert "gradient_boosting" in results
        assert len(results) == 3


class TestGetFeatureImportance:
    def test_returns_dict_with_positive_values(self):
        df = generate_synthetic_churn_data(300)
        importance = get_feature_importance(df, model_name="random_forest")
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())

    def test_sums_to_one(self):
        df = generate_synthetic_churn_data(300)
        importance = get_feature_importance(df, model_name="random_forest")
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_gradient_boosting(self):
        df = generate_synthetic_churn_data(300)
        importance = get_feature_importance(df, model_name="gradient_boosting")
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_logistic_regression(self):
        df = generate_synthetic_churn_data(300)
        importance = get_feature_importance(df, model_name="logistic_regression")
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 1e-6


class TestCrossValidateModels:
    def test_returns_all_three_models(self):
        df = generate_synthetic_churn_data(300)
        results = cross_validate_models(df, n_splits=3)
        assert "logistic_regression" in results
        assert "random_forest" in results
        assert "gradient_boosting" in results

    def test_correct_keys(self):
        df = generate_synthetic_churn_data(300)
        results = cross_validate_models(df, n_splits=3)
        expected_keys = {"mean_accuracy", "std_accuracy", "mean_f1", "std_f1"}
        for model_name, metrics in results.items():
            assert set(metrics.keys()) == expected_keys, f"Missing keys for {model_name}"

    def test_custom_n_splits(self):
        df = generate_synthetic_churn_data(300)
        results = cross_validate_models(df, n_splits=4)
        assert len(results) == 3
        for metrics in results.values():
            assert 0.0 <= metrics["mean_accuracy"] <= 1.0
            assert metrics["std_accuracy"] >= 0.0

    def test_accuracy_in_valid_range(self):
        df = generate_synthetic_churn_data(300)
        results = cross_validate_models(df, n_splits=3)
        for metrics in results.values():
            assert 0.0 <= metrics["mean_accuracy"] <= 1.0
            assert 0.0 <= metrics["mean_f1"] <= 1.0
