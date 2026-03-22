import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1c26 0%, #0e3460 50%, #1a1c26 100%);
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 32px;
        border: 1px solid #2d3561;
    ">
        <h1 style="color:#ffffff;margin:0;font-size:2.6rem;">📉 Customer Churn Prediction</h1>
        <p style="color:#a0aec0;font-size:1.15rem;margin-top:12px;max-width:720px;">
            An interactive, end-to-end machine-learning walkthrough — from raw data to actionable predictions.
            Learn <em>what</em> each step does, <em>why</em> it matters, and <em>how</em> to interpret the results.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── What is Churn? ────────────────────────────────────────────────────────────
st.markdown("## 🤔 What is Customer Churn?")
st.markdown(
    """
    **Customer churn** (also called *customer attrition*) is when a customer **stops doing business** with a company —
    cancelling a subscription, switching to a competitor, or simply going silent.

    > 💬 *"Churn rate = % of customers lost in a given period."*

    In telecommunications, SaaS, banking, and e-commerce, churn is one of the most critical KPIs.
    Acquiring a new customer costs **5–7× more** than retaining an existing one, so even a 1% reduction
    in monthly churn can translate to millions of dollars in saved revenue.
    """
)

with st.expander("📖 Learn More — Business Impact of Churn"):
    st.markdown(
        """
        ### Why Churn Matters

        | Impact | Detail |
        |--------|--------|
        | **Revenue loss** | Every churned customer takes their recurring revenue with them |
        | **Acquisition cost** | CAC (Customer Acquisition Cost) often exceeds 12 months of revenue |
        | **Brand signal** | High churn indicates product-market fit or service-quality problems |
        | **Compounding effect** | Losing 5% monthly means losing ~46% of customers per year |

        ### How ML Helps
        Machine learning lets us **predict** which customers are at risk *before* they churn —
        giving retention teams time to intervene with targeted offers, support outreach, or proactive fixes.

        A well-calibrated churn model turns reactive firefighting into **proactive customer success**.
        """
    )

st.divider()

# ── ML Pipeline ───────────────────────────────────────────────────────────────
st.markdown("## 🔄 The Machine Learning Pipeline")
st.markdown(
    """
    <div style="
        background:#1a1c26;border-radius:12px;padding:24px 32px;
        font-size:1.05rem;letter-spacing:0.02em;text-align:center;
        border:1px solid #2d3561;margin-bottom:16px;
    ">
        📥 <strong>Data Generation</strong>
        &nbsp;→&nbsp;
        🔧 <strong>Preprocessing</strong>
        &nbsp;→&nbsp;
        🧠 <strong>Model Training</strong>
        &nbsp;→&nbsp;
        📊 <strong>Evaluation</strong>
        &nbsp;→&nbsp;
        🎯 <strong>Prediction</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

pipeline_steps = [
    (
        "📥 Step 1 — Data Generation",
        """
        We generate a **synthetic customer dataset** with realistic features:
        age, tenure, monthly charges, contract type, internet service, and payment method.

        The churn label is created using a **logistic function** that combines these features —
        e.g. short-tenure customers on month-to-month contracts with high charges are more likely to churn.

        *In production you'd replace this with real CRM / billing data.*
        """,
    ),
    (
        "🔧 Step 2 — Preprocessing",
        """
        Raw data is rarely model-ready. Preprocessing does two things:

        1. **Label Encoding** — converts text categories (`month`, `year`, `two_year`) into numbers (0, 1, 2)
           so the model can do math on them.
        2. **Standardization** — rescales numeric features (e.g. `monthly_charges` ∈ [20, 200])
           to have **mean = 0, std = 1**.  This prevents large-magnitude features from dominating.

        Result: every feature is on the same playing field.
        """,
    ),
    (
        "🧠 Step 3 — Model Training",
        """
        Three classifiers are trained on 80% of the data:

        - **Logistic Regression** — learns a linear boundary in feature space
        - **Random Forest** — builds 100 decision trees and votes
        - **Gradient Boosting** — builds trees sequentially, each correcting the last

        Training = finding the model parameters that minimise prediction error on the training set.
        """,
    ),
    (
        "📊 Step 4 — Evaluation",
        """
        Models are evaluated on the **held-out 20% test set** (data they never trained on).

        We compute five metrics:  Accuracy, Precision, Recall, F1, and ROC-AUC.
        Each tells a different story — see the Metrics section below for the full breakdown.
        """,
    ),
    (
        "🎯 Step 5 — Prediction",
        """
        The trained model can now score *any* new customer with a **churn probability** between 0 and 1.

        - Score < 0.3 → 🟢 Low risk — continue normal relationship
        - Score 0.3–0.6 → 🟡 Medium risk — flag for proactive outreach
        - Score > 0.6 → 🔴 High risk — immediate retention intervention

        This lets customer-success teams prioritise their limited time and budget.
        """,
    ),
]

for title, body in pipeline_steps:
    with st.expander(title):
        st.markdown(body)

st.divider()

# ── Models We Use ─────────────────────────────────────────────────────────────
st.markdown("## 🤖 Models We Use")

col_lr, col_rf, col_gb = st.columns(3)

with col_lr:
    st.markdown("### 📐 Logistic Regression")
    st.info("A simple, transparent linear classifier — the \"hello world\" of classification.")
    with st.expander("📖 How it works"):
        st.markdown(
            """
            Logistic Regression learns a **weighted sum** of all features, then squashes the result
            through a **sigmoid function** to get a probability between 0 and 1.

            ```
            P(churn) = sigmoid(w₁·tenure + w₂·charges + … + b)
            ```

            The model *learns* the weights `w` during training so that the predictions
            match the actual labels as closely as possible.

            **Intuition:** imagine drawing a straight line (or hyperplane) through feature space
            to separate churners from non-churners. Logistic Regression finds the best line.

            ✅ **Strengths:** fast, interpretable, great baseline  
            ⚠️ **Weaknesses:** assumes linear decision boundary; struggles with complex patterns
            """
        )

with col_rf:
    st.markdown("### 🌲 Random Forest")
    st.info("An ensemble of 100 decision trees that vote together — robust and powerful.")
    with st.expander("📖 How it works"):
        st.markdown(
            """
            A Random Forest builds **100 independent decision trees**, each trained on a
            random subset of the data and random subset of features (*bagging*).

            **Intuition:** imagine asking 100 experts who each only know part of the picture.
            Their majority vote is usually more reliable than any single expert.

            Each tree splits data based on feature thresholds:
            - Is `tenure < 12`? → go left (higher churn risk)
            - Is `contract_type == two_year`? → go right (lower churn risk)

            Trees are grown deep and then averaged, which **reduces overfitting**.

            ✅ **Strengths:** handles non-linear patterns, robust to outliers, provides feature importance  
            ⚠️ **Weaknesses:** slower to train, harder to interpret individual trees
            """
        )

with col_gb:
    st.markdown("### 🚀 Gradient Boosting")
    st.info("Builds trees sequentially — each new tree fixes the errors of the previous ones.")
    with st.expander("📖 How it works"):
        st.markdown(
            """
            Gradient Boosting is a **sequential ensemble** method.  It starts with a simple model,
            checks where it went wrong, and trains the *next* tree specifically to fix those mistakes.

            **Intuition:** like a student who reviews their wrong exam answers and studies only
            those topics for the next test. Each iteration gets smarter about past mistakes.

            The key insight is that we're **optimising a loss function** — minimising prediction
            error by stepping in the direction of the gradient (hence the name).

            ✅ **Strengths:** often the most accurate, handles mixed data types well  
            ⚠️ **Weaknesses:** slower to train, prone to overfitting if not tuned carefully
            """
        )

st.divider()

# ── Evaluation Metrics ────────────────────────────────────────────────────────
st.markdown("## 📏 Evaluation Metrics — Explained")

st.markdown(
    """
    > **Context:** Imagine our test set has 200 customers, 40 of whom actually churned.
    > Our model predicts 50 will churn.  Of those 50, 30 actually did churn and 20 did not.
    """
)

metrics = [
    {
        "name": "✅ Accuracy",
        "formula": "(TP + TN) / Total",
        "plain": "What fraction of *all* predictions were correct?",
        "churn": "With only ~20-30% churners in a typical dataset, a model that always predicts 'no churn' gets 75% accuracy for free. **Misleading on imbalanced data!**",
        "example": "Model got 170/200 right → Accuracy = 85%",
    },
    {
        "name": "🎯 Precision",
        "formula": "TP / (TP + FP)",
        "plain": "Of all customers we *flagged* as churners, how many actually churned?",
        "churn": "High precision means fewer wasted retention offers sent to customers who weren't going to leave. Matters when intervention is expensive.",
        "example": "Of 50 flagged: 30 true churners → Precision = 30/50 = 60%",
    },
    {
        "name": "📡 Recall",
        "formula": "TP / (TP + FN)",
        "plain": "Of all customers who *actually* churned, how many did we catch?",
        "churn": "High recall means fewer at-risk customers slip through undetected. Matters when missing a churner is very costly.",
        "example": "Caught 30 of 40 real churners → Recall = 30/40 = 75%",
    },
    {
        "name": "⚖️ F1 Score",
        "formula": "2 × (Precision × Recall) / (Precision + Recall)",
        "plain": "The harmonic mean of Precision and Recall — a single balanced score.",
        "churn": "F1 is the go-to metric for imbalanced datasets. It rewards models that are *both* precise and thorough. A model can't game it by just predicting everything.",
        "example": "Precision=0.60, Recall=0.75 → F1 = 2×(0.60×0.75)/(0.60+0.75) ≈ **0.667**",
    },
    {
        "name": "📈 ROC-AUC",
        "formula": "Area Under the ROC Curve",
        "plain": "How well does the model *rank* churners above non-churners across all thresholds?",
        "churn": "AUC = 1.0 means perfect ranking; AUC = 0.5 means random guessing. It's threshold-independent — great for comparing models before you decide on a cutoff.",
        "example": "AUC = 0.85 means: if you pick a random churner and a random non-churner, the model scores the churner higher 85% of the time.",
    },
]

for m in metrics:
    with st.expander(m["name"]):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"**Formula:**  `{m['formula']}`")
            st.markdown(f"**In plain English:** {m['plain']}")
        with c2:
            st.info(f"💡 **Why it matters for churn:** {m['churn']}")
        st.markdown(f"📌 **Example:** {m['example']}")

st.divider()
st.markdown(
    "<p style='color:#666;text-align:center;'>Navigate to <strong>Train & Evaluate</strong> to run the pipeline interactively →</p>",
    unsafe_allow_html=True,
)
