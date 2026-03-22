import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from src.churn_model import generate_synthetic_churn_data, preprocess, train_and_evaluate

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1c26 0%, #0e3460 50%, #1a1c26 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        border: 1px solid #2d3561;
    ">
        <h1 style="color:#ffffff;margin:0;font-size:2.2rem;">📈 Train & Evaluate Models</h1>
        <p style="color:#a0aec0;font-size:1.05rem;margin-top:10px;">
            Walk through the full ML pipeline step by step — generate data, preprocess it,
            train three models, and compare their results.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Step indicator ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex;gap:12px;margin-bottom:28px;flex-wrap:wrap;">
        <div style="background:#0e3460;border-radius:8px;padding:10px 18px;color:#90cdf4;font-weight:600;">① Generate Data</div>
        <div style="color:#4a5568;padding:10px 4px;">→</div>
        <div style="background:#1a1c26;border-radius:8px;padding:10px 18px;color:#718096;border:1px solid #2d3561;">② Preprocess</div>
        <div style="color:#4a5568;padding:10px 4px;">→</div>
        <div style="background:#1a1c26;border-radius:8px;padding:10px 18px;color:#718096;border:1px solid #2d3561;">③ Train</div>
        <div style="color:#4a5568;padding:10px 4px;">→</div>
        <div style="background:#1a1c26;border-radius:8px;padding:10px 18px;color:#718096;border:1px solid #2d3561;">④ Evaluate</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Step 1: Generate Data ─────────────────────────────────────────────────────
st.markdown("## 📥 Step 1 — Generate Data")

n_samples = st.slider(
    "Number of synthetic customers",
    min_value=200,
    max_value=5000,
    value=1000,
    step=100,
    help="More samples = more robust training, but slower. Try 1000 as a starting point.",
)

if st.button("🎲 Generate Dataset", type="primary"):
    with st.spinner("Generating synthetic customer data..."):
        df = generate_synthetic_churn_data(n_samples)
    st.session_state["df"] = df
    st.success(f"✅ Dataset generated! {n_samples:,} customers created.")

if "df" in st.session_state:
    df = st.session_state["df"]
    churn_count = int(df["churn"].sum())
    no_churn_count = n_samples - churn_count
    churn_rate = churn_count / n_samples * 100

    # Key stats
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Total Customers", f"{n_samples:,}")
    with sc2:
        st.metric("Churned", f"{churn_count:,}", delta=f"{churn_rate:.1f}%")
    with sc3:
        st.metric("Features", "8 columns")

    st.markdown("### 👀 Dataset Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 📊 Churn Distribution")
    chart_data = pd.DataFrame(
        {"Count": [no_churn_count, churn_count]},
        index=["Did Not Churn", "Churned"],
    )
    st.bar_chart(chart_data)

    with st.expander("📖 How is synthetic data generated?"):
        st.markdown(
            f"""
            ### Synthetic Data Generation

            We use `numpy.random` to create {n_samples:,} synthetic customers with realistic feature distributions:

            | Feature | Type | Distribution |
            |---------|------|-------------|
            | `age` | Numeric | Uniform integers [18, 70] |
            | `tenure` | Numeric | Uniform integers [1, 72] months |
            | `monthly_charges` | Numeric | Normal(μ=65, σ=30), clipped at $20 |
            | `total_charges` | Numeric | Normal(μ=2000, σ=1500), clipped at $0 |
            | `contract_type` | Categorical | Uniform choice: month / year / two_year |
            | `internet_service` | Categorical | Uniform choice: DSL / Fiber / No |
            | `payment_method` | Categorical | Uniform choice: auto / check / electronic |

            ### How the Churn Label is Created

            The churn label is generated using a **logistic function** (the same one logistic regression uses!):

            ```
            churn_probability = sigmoid(
                monthly_charges / 100 - 0.5
                + 1.5  if tenure < 12 months
                - 2.0  if contract_type == two_year
            )
            ```

            - Short-tenure customers are **more likely to churn** (no loyalty built yet)
            - High monthly charges **increase churn probability** (financial pressure)
            - Two-year contracts **strongly reduce churn** (customers are committed)

            Each customer is then randomly assigned churn = 1 with their computed probability.
            This ensures the label is **realistic but stochastic** — not deterministic.
            """
        )

    st.divider()

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    st.markdown("## 🔧 Step 2 — Preprocessing")
    st.info(
        "Preprocessing converts raw data into a format models can learn from. "
        "This step is automatic — no configuration needed."
    )

    with st.expander("📖 What is preprocessing?"):
        st.markdown(
            """
            ### Two Key Transformations

            #### 1. Label Encoding (for categorical features)
            Machine learning models work with *numbers*, not strings.
            Label encoding maps each category to an integer:

            ```
            month     → 0
            two_year  → 1
            year      → 2
            ```

            This is fine for tree-based models (Random Forest, Gradient Boosting).
            For Logistic Regression, one-hot encoding is sometimes preferred, but label encoding
            works here because our dataset is relatively simple.

            #### 2. StandardScaler (for numeric features)
            Raw numeric features can have wildly different scales:
            - `tenure`: 1 → 72
            - `total_charges`: 0 → 10,000+

            Without scaling, a model might give `total_charges` **100× more influence** than `tenure`
            just because its numbers are bigger. StandardScaler rescales each feature to:

            ```
            z = (x - mean) / std_deviation
            ```

            After scaling: **mean = 0, std = 1** for every feature. Fair competition!
            """
        )

    # Before/After sample
    st.markdown("### Before vs After Preprocessing")
    raw_sample = df.drop(columns=["customer_id", "churn"]).head(3)
    X_proc, _, _ = preprocess(df)

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Raw Data (first 3 rows)**")
        st.dataframe(raw_sample, use_container_width=True)
    with col_after:
        st.markdown("**After Preprocessing (scaled numerics)**")
        import numpy as np
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("churn", "customer_id")]
        processed_df = pd.DataFrame(X_proc[:3], columns=num_cols + cat_cols).round(3)
        st.dataframe(processed_df, use_container_width=True)

    st.divider()

    # ── Step 3: Train Models ──────────────────────────────────────────────────
    st.markdown("## 🧠 Step 3 — Train Models")

    with st.expander("📖 How does each model learn?"):
        st.markdown(
            """
            ### Logistic Regression
            Finds the optimal **weights** for each feature by minimising log-loss (cross-entropy).
            Uses gradient descent to iteratively adjust weights until convergence.
            Training is **fast** (~milliseconds for 1000 samples).

            ### Random Forest
            Builds **100 decision trees** in parallel, each on a bootstrapped sample of the data
            and a random subset of features. Each tree votes, and the majority wins.
            Training is **parallelisable** — moderately fast.

            ### Gradient Boosting
            Builds trees **sequentially**. Each new tree is trained on the *residual errors*
            of the previous ensemble — literally chasing its own mistakes.
            More computationally expensive, but often the most accurate.

            ---
            All models use an **80/20 train/test split** — 80% of data for learning,
            20% held out for evaluation (models never see this during training).
            """
        )

    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training Logistic Regression, Random Forest, and Gradient Boosting..."):
            results = train_and_evaluate(st.session_state["df"])
        st.session_state["results"] = results
        st.success("✅ All three models trained successfully!")

    st.divider()

    # ── Step 4: View Results ──────────────────────────────────────────────────
    if "results" in st.session_state:
        st.markdown("## 📊 Step 4 — Evaluation Results")

        results = st.session_state["results"]
        model_display = {
            "logistic_regression": ("📐 Logistic Regression", "#4299e1"),
            "random_forest": ("🌲 Random Forest", "#48bb78"),
            "gradient_boosting": ("🚀 Gradient Boosting", "#ed8936"),
        }
        metrics_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_labels = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
            "roc_auc": "ROC-AUC",
        }

        # ── Metric cards ──────────────────────────────────────────────────────
        st.markdown("### 🏆 Model Comparison")
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]

        for i, (model_key, (display_name, color)) in enumerate(model_display.items()):
            with columns[i]:
                st.markdown(
                    f"<h4 style='color:{color};margin-bottom:8px;'>{display_name}</h4>",
                    unsafe_allow_html=True,
                )
                m = results[model_key]
                for metric_key in metrics_order:
                    val = m[metric_key]
                    label = metric_labels[metric_key]
                    st.metric(label=label, value=f"{val:.4f}")

        # ── Best model per metric ─────────────────────────────────────────────
        st.markdown("### 🥇 Best Model Per Metric")
        for metric_key in metrics_order:
            best_model = max(results, key=lambda k: results[k][metric_key])
            best_val = results[best_model][metric_key]
            display_name = model_display[best_model][0]
            label = metric_labels[metric_key]
            st.success(f"**{label}** — Best: {display_name} with **{best_val:.4f}**")

        # ── Comparison bar chart ──────────────────────────────────────────────
        st.markdown("### 📈 Side-by-Side Metrics Chart")
        chart_df = pd.DataFrame(results).T.rename(columns=metric_labels)
        chart_df.index = [model_display[k][0] for k in chart_df.index]
        st.bar_chart(chart_df)

        # ── Metric explanations ───────────────────────────────────────────────
        with st.expander("💡 What do these metrics mean?"):
            st.markdown(
                """
                | Metric | One-line definition | Churn relevance |
                |--------|---------------------|-----------------|
                | **Accuracy** | Correct predictions / total predictions | Can be misleading with imbalanced classes |
                | **Precision** | True positives / all predicted positives | "How many flagged customers actually churned?" |
                | **Recall** | True positives / all actual positives | "How many real churners did we catch?" |
                | **F1 Score** | Harmonic mean of Precision & Recall | Best single metric for imbalanced churn data |
                | **ROC-AUC** | Ranking quality across all thresholds | "How well does the model separate churners?" |

                ### Precision vs Recall Tradeoff
                - **High precision, low recall** → conservative model: only flags obvious churners, misses many
                - **High recall, low precision** → aggressive model: catches most churners, but many false alarms
                - **F1** balances both. For churn, **recall is often prioritised** — missing a churner is expensive.
                """
            )

        with st.expander("📖 What is ROC-AUC and why does it matter?"):
            st.markdown(
                """
                ### ROC Curve

                The **Receiver Operating Characteristic (ROC) curve** plots:
                - **Y-axis:** True Positive Rate (Recall) — how many real churners we catch
                - **X-axis:** False Positive Rate — how many non-churners we incorrectly flag

                Each point on the curve represents a different **classification threshold**.
                Setting threshold = 0.3 catches more churners (high recall) but also more false alarms.
                Setting threshold = 0.7 is more conservative (high precision) but misses real churners.

                ### AUC — Area Under the Curve

                - **AUC = 1.0** → perfect model
                - **AUC = 0.5** → random guessing (coin flip)
                - **AUC = 0.85** → "if I pick a random churner and a random non-churner,
                  my model scores the churner higher 85% of the time"

                AUC is **threshold-independent** — it evaluates the model's *ranking* ability,
                making it ideal for comparing models before you've decided on a deployment threshold.
                """
            )

        # ── Full results table ────────────────────────────────────────────────
        st.markdown("### 📋 Full Results Table")
        results_table = pd.DataFrame(results).T
        results_table.index = [model_display[k][0] for k in results_table.index]
        results_table.columns = [metric_labels[c] for c in results_table.columns]
        st.dataframe(results_table.style.format("{:.4f}").highlight_max(axis=0, color="#1a4a2e"), use_container_width=True)

    elif "df" in st.session_state:
        st.info("👆 Click **Train All Models** above to see evaluation results here.")

else:
    st.warning("👆 Start by generating a dataset using the **Generate Dataset** button above.")
