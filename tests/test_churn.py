"""
Tests for the churn prediction model module.

Covers:
    - Synthetic data generation (shape, types, reproducibility, edge cases)
    - Preprocessing (scaling, encoding, validation)
    - Training and evaluation (metric ranges, model completeness)
    - Feature importance (normalization, all model types)
    - Cross-validation (fold configuration, metric validity)
    - Confusion matrices (shape, value consistency)
    - Single customer prediction (risk levels, edge cases)
    - Data validation (error handling for malformed inputs)
    - Model persistence (save/load round-trip)
"""

import pytest
import numpy as np
import pandas as pd
from src.churn_model import (
    generate_synthetic_churn_data,
    preprocess,
    train_and_evaluate,
    get_feature_importance,
    cross_validate_models,
    get_confusion_matrices,
    predict_single_customer,
    validate_dataframe,
    save_model,
    load_model,
    find_optimal_threshold,
    compute_learning_curve,
    SUPPORTED_MODELS,
    EXPECTED_COLUMNS,
)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_shape(self):
        df = generate_synthetic_churn_data(100)
        assert df.shape == (100, 9)

    def test_has_churn(self):
        assert "churn" in generate_synthetic_churn_data().columns

    def test_reproducible(self):
        """Same seed must produce identical datasets."""
        df1 = generate_synthetic_churn_data(seed=42)
        df2 = generate_synthetic_churn_data(seed=42)
        assert df1.equals(df2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different datasets."""
        df1 = generate_synthetic_churn_data(200, seed=42)
        df2 = generate_synthetic_churn_data(200, seed=99)
        assert not df1.equals(df2)

    def test_correct_columns(self):
        df = generate_synthetic_churn_data(50)
        assert set(df.columns) == EXPECTED_COLUMNS

    def test_churn_values_binary(self):
        df = generate_synthetic_churn_data(500)
        assert set(df["churn"].unique()).issubset({0, 1})

    def test_monthly_charges_clipped(self):
        """Monthly charges should be clipped at minimum of $20."""
        df = generate_synthetic_churn_data(5000)
        assert df["monthly_charges"].min() >= 20

    def test_total_charges_non_negative(self):
        """Total charges should never be negative."""
        df = generate_synthetic_churn_data(5000)
        assert df["total_charges"].min() >= 0

    def test_minimum_samples(self):
        """Should work with just 1 sample."""
        df = generate_synthetic_churn_data(1)
        assert len(df) == 1

    def test_invalid_n_samples_raises(self):
        """n_samples < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            generate_synthetic_churn_data(0)

    def test_contract_types(self):
        """All expected contract types should appear in large datasets."""
        df = generate_synthetic_churn_data(5000)
        expected = {"month", "year", "two_year"}
        assert set(df["contract_type"].unique()) == expected

    def test_internet_service_types(self):
        df = generate_synthetic_churn_data(5000)
        expected = {"DSL", "Fiber", "No"}
        assert set(df["internet_service"].unique()) == expected

    def test_customer_ids_unique(self):
        df = generate_synthetic_churn_data(500)
        assert df["customer_id"].nunique() == 500


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_shapes(self):
        df = generate_synthetic_churn_data(200)
        X, y, _ = preprocess(df)
        assert X.shape[0] == 200
        assert y.shape[0] == 200

    def test_feature_count(self):
        """Should have 7 features (4 numeric + 3 categorical, excluding id and churn)."""
        df = generate_synthetic_churn_data(100)
        X, _, _ = preprocess(df)
        assert X.shape[1] == 7

    def test_scaling_zero_mean(self):
        """After StandardScaler, features should have approximately zero mean."""
        df = generate_synthetic_churn_data(1000)
        X, _, _ = preprocess(df)
        means = np.abs(X.mean(axis=0))
        assert all(means < 0.1), f"Feature means too far from 0: {means}"

    def test_scaler_returned(self):
        """Preprocessing must return a fitted StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        df = generate_synthetic_churn_data(100)
        _, _, scaler = preprocess(df)
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mean_")  # fitted scaler has mean_ attribute


# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------

class TestValidateDataframe:
    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe(pd.DataFrame())

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"age": [25], "churn": [0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df)

    def test_non_binary_churn_raises(self):
        df = generate_synthetic_churn_data(100)
        df.loc[0, "churn"] = 2  # invalid value
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            validate_dataframe(df)

    def test_valid_dataframe_passes(self):
        df = generate_synthetic_churn_data(100)
        validate_dataframe(df)  # should not raise


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------

class TestTrainEvaluate:
    def test_returns_results(self):
        df = generate_synthetic_churn_data(200)
        results = train_and_evaluate(df)
        assert "logistic_regression" in results
        assert "roc_auc" in results["logistic_regression"]

    def test_returns_three_models(self):
        df = generate_synthetic_churn_data(300)
        results = train_and_evaluate(df)
        assert set(results.keys()) == set(SUPPORTED_MODELS)

    def test_metrics_in_valid_range(self):
        """All metrics should be between 0 and 1."""
        df = generate_synthetic_churn_data(500)
        results = train_and_evaluate(df)
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                assert 0.0 <= value <= 1.0, (
                    f"{model_name}.{metric_name} = {value} is out of [0, 1]"
                )

    def test_invalid_test_size_raises(self):
        df = generate_synthetic_churn_data(200)
        with pytest.raises(ValueError, match="test_size must be between"):
            train_and_evaluate(df, test_size=0.0)
        with pytest.raises(ValueError, match="test_size must be between"):
            train_and_evaluate(df, test_size=1.0)

    def test_custom_test_size(self):
        """Different test sizes should produce valid results."""
        df = generate_synthetic_churn_data(500)
        results = train_and_evaluate(df, test_size=0.3)
        assert len(results) == 3
        for metrics in results.values():
            assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_all_expected_metrics(self):
        """Each model should report exactly 5 metrics."""
        df = generate_synthetic_churn_data(300)
        results = train_and_evaluate(df)
        expected_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        for model_name, metrics in results.items():
            assert set(metrics.keys()) == expected_metrics


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

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

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
    def test_all_features_present(self, model_name):
        """Importance dict should contain all 7 features for every model."""
        df = generate_synthetic_churn_data(300)
        importance = get_feature_importance(df, model_name=model_name)
        assert len(importance) == 7

    def test_invalid_model_raises(self):
        df = generate_synthetic_churn_data(100)
        with pytest.raises(ValueError, match="Unknown model_name"):
            get_feature_importance(df, model_name="neural_network")


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

class TestCrossValidateModels:
    def test_returns_all_three_models(self):
        df = generate_synthetic_churn_data(300)
        results = cross_validate_models(df, n_splits=3)
        assert set(results.keys()) == set(SUPPORTED_MODELS)

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

    def test_std_is_non_negative(self):
        """Standard deviation must always be >= 0."""
        df = generate_synthetic_churn_data(500)
        results = cross_validate_models(df, n_splits=5)
        for metrics in results.values():
            assert metrics["std_accuracy"] >= 0
            assert metrics["std_f1"] >= 0


# ---------------------------------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------------------------------

class TestConfusionMatrices:
    def test_returns_all_models(self):
        df = generate_synthetic_churn_data(300)
        results = get_confusion_matrices(df)
        assert set(results.keys()) == set(SUPPORTED_MODELS)

    def test_matrix_shape(self):
        """Confusion matrix should be 2x2."""
        df = generate_synthetic_churn_data(300)
        results = get_confusion_matrices(df)
        for model_name, result in results.items():
            matrix = result["matrix"]
            assert len(matrix) == 2
            assert len(matrix[0]) == 2
            assert len(matrix[1]) == 2

    def test_values_sum_to_test_size(self):
        """TN + FP + FN + TP should equal the number of test samples."""
        n_samples = 500
        test_size = 0.2
        df = generate_synthetic_churn_data(n_samples)
        results = get_confusion_matrices(df, test_size=test_size)
        expected_test = int(n_samples * test_size)

        for model_name, result in results.items():
            total = result["tn"] + result["fp"] + result["fn"] + result["tp"]
            assert total == expected_test, (
                f"{model_name}: confusion matrix sums to {total}, expected {expected_test}"
            )

    def test_all_values_non_negative(self):
        df = generate_synthetic_churn_data(300)
        results = get_confusion_matrices(df)
        for result in results.values():
            assert result["tn"] >= 0
            assert result["fp"] >= 0
            assert result["fn"] >= 0
            assert result["tp"] >= 0


# ---------------------------------------------------------------------------
# Single Customer Prediction
# ---------------------------------------------------------------------------

class TestPredictSingleCustomer:
    @pytest.fixture
    def training_data(self):
        return generate_synthetic_churn_data(500)

    @pytest.fixture
    def high_risk_customer(self):
        """Young customer, short tenure, month-to-month, high charges."""
        return {
            "age": 22, "tenure": 2, "monthly_charges": 120,
            "total_charges": 240, "contract_type": "month",
            "internet_service": "Fiber", "payment_method": "electronic",
        }

    @pytest.fixture
    def low_risk_customer(self):
        """Established customer, two-year contract, auto-pay."""
        return {
            "age": 55, "tenure": 60, "monthly_charges": 30,
            "total_charges": 1800, "contract_type": "two_year",
            "internet_service": "DSL", "payment_method": "auto",
        }

    def test_returns_expected_keys(self, training_data, high_risk_customer):
        result = predict_single_customer(
            high_risk_customer, training_data, "gradient_boosting"
        )
        assert "churn_probability" in result
        assert "risk_level" in result
        assert "risk_emoji" in result
        assert "model_used" in result

    def test_probability_in_range(self, training_data, high_risk_customer):
        result = predict_single_customer(
            high_risk_customer, training_data, "random_forest"
        )
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_risk_levels_valid(self, training_data, high_risk_customer):
        result = predict_single_customer(
            high_risk_customer, training_data, "gradient_boosting"
        )
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert result["risk_emoji"] in ("🟢", "🟡", "🔴")

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
    def test_all_models_work(self, training_data, high_risk_customer, model_name):
        """Single-customer prediction should work with all supported models."""
        result = predict_single_customer(
            high_risk_customer, training_data, model_name
        )
        assert result["model_used"] == model_name
        assert 0.0 <= result["churn_probability"] <= 1.0


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        """Model should produce same predictions after save/load."""
        df = generate_synthetic_churn_data(300)
        output_dir = str(tmp_path / "models")

        save_model(df, model_name="random_forest", output_dir=output_dir)
        model, scaler, metadata = load_model(
            model_name="random_forest", model_dir=output_dir
        )

        assert metadata["model_name"] == "random_forest"
        assert metadata["n_training_samples"] == 300
        assert len(metadata["feature_names"]) == 7

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model(model_name="random_forest", model_dir=str(tmp_path))

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
    def test_all_models_saveable(self, tmp_path, model_name):
        """Every supported model should be saveable and loadable."""
        df = generate_synthetic_churn_data(200)
        output_dir = str(tmp_path / model_name)

        save_model(df, model_name=model_name, output_dir=output_dir)
        model, scaler, metadata = load_model(
            model_name=model_name, model_dir=output_dir
        )
        assert metadata["model_name"] == model_name


# ---------------------------------------------------------------------------
# Threshold Optimization
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
    def test_returns_expected_keys(self):
        df = generate_synthetic_churn_data(300)
        result = find_optimal_threshold(df, "gradient_boosting", metric="f1")
        assert "optimal_threshold" in result
        assert "best_score" in result
        assert "metric" in result
        assert "model_name" in result
        assert "threshold_scores" in result

    def test_threshold_in_valid_range(self):
        """Optimal threshold must be between 0.05 and 0.95."""
        df = generate_synthetic_churn_data(300)
        result = find_optimal_threshold(df, "random_forest", metric="f1")
        assert 0.05 <= result["optimal_threshold"] <= 0.95

    def test_best_score_in_valid_range(self):
        df = generate_synthetic_churn_data(300)
        result = find_optimal_threshold(df, "gradient_boosting", metric="f1")
        assert 0.0 <= result["best_score"] <= 1.0

    @pytest.mark.parametrize("metric", ["f1", "precision", "recall", "accuracy"])
    def test_all_metrics_supported(self, metric):
        """All four metrics should work without errors."""
        df = generate_synthetic_churn_data(300)
        result = find_optimal_threshold(df, "gradient_boosting", metric=metric)
        assert result["metric"] == metric
        assert 0.0 <= result["best_score"] <= 1.0

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
    def test_all_models_supported(self, model_name):
        """Threshold optimization should work with all model types."""
        df = generate_synthetic_churn_data(300)
        result = find_optimal_threshold(df, model_name, metric="f1")
        assert result["model_name"] == model_name

    def test_threshold_scores_length(self):
        """Should have scores for each threshold step."""
        df = generate_synthetic_churn_data(300)
        result = find_optimal_threshold(df, "gradient_boosting", metric="f1")
        # 0.05 to 0.95 in steps of 0.01 = 91 points
        assert len(result["threshold_scores"]) == 91

    def test_invalid_metric_raises(self):
        df = generate_synthetic_churn_data(100)
        with pytest.raises(ValueError, match="Invalid metric"):
            find_optimal_threshold(df, "gradient_boosting", metric="auc")


# ---------------------------------------------------------------------------
# Learning Curve
# ---------------------------------------------------------------------------

class TestComputeLearningCurve:
    def test_returns_expected_keys(self):
        df = generate_synthetic_churn_data(300)
        result = compute_learning_curve(df, "gradient_boosting", n_points=4)
        assert "train_sizes" in result
        assert "train_scores" in result
        assert "test_scores" in result
        assert "model_name" in result

    def test_correct_number_of_points(self):
        df = generate_synthetic_churn_data(500)
        result = compute_learning_curve(df, "random_forest", n_points=6)
        assert len(result["train_sizes"]) == 6
        assert len(result["train_scores"]) == 6
        assert len(result["test_scores"]) == 6

    def test_scores_in_valid_range(self):
        df = generate_synthetic_churn_data(500)
        result = compute_learning_curve(df, "gradient_boosting", n_points=4)
        for score in result["train_scores"] + result["test_scores"]:
            assert 0.0 <= score <= 1.0

    def test_train_sizes_increasing(self):
        """Training sizes should be monotonically increasing."""
        df = generate_synthetic_churn_data(500)
        result = compute_learning_curve(df, "gradient_boosting", n_points=5)
        sizes = result["train_sizes"]
        assert all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1))

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
    def test_all_models_supported(self, model_name):
        """Learning curve should work with all model types."""
        df = generate_synthetic_churn_data(300)
        result = compute_learning_curve(df, model_name, n_points=3)
        assert result["model_name"] == model_name
        assert len(result["train_scores"]) == 3
