"""
Customer Satisfaction Ticket Prediction — Streamlit app
=========================================================
Interactive front-end for the analysis in src/Customer_Support.py.

Pages:
  - Overview            : dataset summary + headline finding
  - Data Explorer        : interactive EDA on the closed / rated tickets
  - Model Training & Results : trains Logistic Regression, Random Forest,
                            SGD, compares them, shows confusion matrices,
                            feature importance, and 5-fold CV
  - Predict a Ticket     : live "what would this model guess" form
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error,
)

sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Satisfaction Ticket Prediction",
    page_icon="🎫",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "data" / "customer_support_dataset.csv"

POSITIVE_WORDS = {
    "good", "great", "excellent", "happy", "satisfied", "resolved", "fixed", "helpful", "quick",
    "easy", "love", "thank", "thanks", "amazing", "perfect", "wonderful", "best", "pleasant",
    "efficient", "friendly", "win", "success", "benefit", "positive",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "angry", "frustrated", "disappointed", "broken", "issue", "problem", "delay",
    "slow", "poor", "worst", "annoyed", "difficult", "failed", "error", "complain", "unhappy",
    "waste", "horrible", "fail", "negative", "wrong",
}

ONEHOT_COLS = ["Customer Gender", "Ticket Type", "Ticket Priority", "Ticket Channel",
               "Type of Customer", "Response"]


def sentiment_score(text: str) -> float:
    words = re.findall(r"[a-zA-Z']+", str(text).lower())
    if not words:
        return 0.0
    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    return (pos - neg) / len(words)


def bucket_age(age: int) -> str:
    if age <= 30:
        return "Young Customer"
    elif 30 < age < 55:
        return "Middle Age Customer"
    return "Old Customer"


# ----------------------------------------------------------------------------
# Data loading & cleaning (mirrors src/Customer_Support.py sections 2-6)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"], format="%Y-%m-%d")
    return df


@st.cache_data(show_spinner=False)
def build_rated_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filter down to closed, rated tickets and engineer the same features as the script."""
    df1 = df_raw.drop(columns=["Ticket ID"]).copy()

    df1["Resolution"] = df1["Resolution"].fillna("None")
    has_response = df1["First Response Time"].notna()
    df1["Customer Satisfaction Rating"] = df1["Customer Satisfaction Rating"]

    df1["Type of Customer"] = df1["Customer Age"].apply(bucket_age)
    df1["Response"] = np.where(has_response, "Yes", "No")

    # Keep only rows that actually have a satisfaction rating (closed tickets)
    df1 = df1[df1["Customer Satisfaction Rating"].notna()].copy()
    df1["Customer Satisfaction Rating"] = df1["Customer Satisfaction Rating"].astype(int)

    df1["Resolution Sentiment"] = df1["Resolution"].apply(sentiment_score)
    df1["Purchase_Month"] = df1["Date of Purchase"].dt.month
    df1["Purchase_DayOfWeek"] = df1["Date of Purchase"].dt.dayofweek

    return df1


@st.cache_data(show_spinner=False)
def build_feature_matrix(df1: pd.DataFrame):
    """Recreate the encoding pipeline from the script; returns X, y, product-frequency map."""
    drop_cols = ["Customer Name", "Customer Email", "Ticket Description", "Ticket Subject",
                 "Resolution", "Date of Purchase", "First Response Time", "Time to Resolution",
                 "Ticket Status"]
    df2 = df1.drop(columns=drop_cols).copy()

    product_freq = df2["Product Purchased"].value_counts(normalize=True)
    df2["Product Purchased Freq"] = df2["Product Purchased"].map(product_freq)

    df2 = pd.get_dummies(df2, columns=ONEHOT_COLS, drop_first=False)

    X = df2.drop(columns=["Customer Satisfaction Rating", "Product Purchased"])
    y = df2["Customer Satisfaction Rating"]

    return X, y, product_freq


@st.cache_resource(show_spinner=False)
def train_all_models(X: pd.DataFrame, y: pd.Series):
    """Train/test split, scaling, and fit all three models + a tuned RF grid search."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    # Random Forest (tuned)
    rf_param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_param_grid, cv=3, scoring="f1_weighted", n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    rf = rf_grid.best_estimator_

    # SGD
    sgd = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    sgd.fit(X_train_scaled, y_train)

    models = {
        "Logistic Regression": {"model": log_reg, "scaled": True},
        "Random Forest": {"model": rf, "scaled": False},
        "SGD (Gradient Descent)": {"model": sgd, "scaled": True},
    }

    return {
        "models": models,
        "scaler": scaler,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "rf_best_params": rf_grid.best_params_,
    }


def evaluate_model(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Within +/-1": (np.abs(np.array(y_true) - np.array(y_pred)) <= 1).mean(),
    }


@st.cache_data(show_spinner=False)
def compute_results_table(_bundle, y_test_hash):
    bundle = _bundle
    rows = []
    preds = {}
    for name, info in bundle["models"].items():
        X_te = bundle["X_test_scaled"] if info["scaled"] else bundle["X_test"]
        y_pred = info["model"].predict(X_te)
        preds[name] = y_pred
        row = evaluate_model(bundle["y_test"], y_pred)
        row["Model"] = name
        rows.append(row)
    results = pd.DataFrame(rows).set_index("Model").sort_values("F1 Score", ascending=False)
    return results, preds


@st.cache_data(show_spinner=False)
def compute_cv_results(_bundle, rf_params, cache_key):
    bundle = _bundle
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model, use_scaled in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), True),
        ("Random Forest", RandomForestClassifier(n_estimators=rf_params["n_estimators"],
                                                  max_depth=rf_params["max_depth"],
                                                  random_state=42, n_jobs=-1), False),
        ("SGD (Gradient Descent)", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42), True),
    ]:
        data = bundle["X_train_scaled"] if use_scaled else bundle["X_train"]
        scores = cross_validate(model, data, bundle["y_train"], cv=cv, scoring=["accuracy", "f1_weighted"])
        rows.append({
            "Model": name,
            "CV Accuracy (mean)": scores["test_accuracy"].mean(),
            "CV Accuracy (std)": scores["test_accuracy"].std(),
            "CV F1 (mean)": scores["test_f1_weighted"].mean(),
            "CV F1 (std)": scores["test_f1_weighted"].std(),
        })
    return pd.DataFrame(rows).set_index("Model")


# ----------------------------------------------------------------------------
# Load everything once
# ----------------------------------------------------------------------------
df_raw = load_raw_data()
df1 = build_rated_dataset(df_raw)
X, y, product_freq = build_feature_matrix(df1)

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🎫 Ticket Satisfaction")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 Data Explorer", "🤖 Model Training & Results", "🎯 Predict a Ticket"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Explores 8,469 customer support tickets and tests whether ticket, customer, "
    "and product attributes can predict a closed ticket's satisfaction rating (1–5)."
)

# ============================================================================
# PAGE 1 — Overview
# ============================================================================
if page == "🏠 Overview":
    st.title("Customer Support Ticket Analysis & Satisfaction Prediction")
    st.write(
        "Explores a customer support ticket dataset and builds classification models to "
        "predict **Customer Satisfaction Rating** (1–5) from ticket, customer, and product "
        "attributes — with the goal of flagging tickets at risk of a poor rating before they close."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tickets", f"{df_raw.shape[0]:,}")
    c2.metric("Rated (closed) tickets", f"{df1.shape[0]:,}")
    c3.metric("Distinct products", f"{df_raw['Product Purchased'].nunique()}")
    c4.metric("Features used in model", f"{X.shape[1]}")

    st.markdown("### Approach")
    st.markdown(
        """
1. **EDA** — shape, dtypes, missing values, duplicates, age/response distributions
2. **Target selection** — `Customer Satisfaction Rating` is only populated for closed tickets
   (2,769 of 8,469 rows); the dataset is filtered to those rows
3. **NLP experiment** — a lexicon-based sentiment score computed on the free-text `Resolution`
   field and tested as a feature
4. **Feature engineering & encoding** — one-hot encoding for low-cardinality categoricals,
   frequency-encoding for the 42-value `Product Purchased` column, numeric scaling
5. **Modeling** — Logistic Regression, Random Forest (tuned via `GridSearchCV`), and an
   `SGDClassifier` (gradient descent), plus a from-scratch batch gradient descent
   implementation for a binary version of the target
6. **Evaluation** — accuracy, precision, recall, F1 (weighted), MAE, "within ±1" accuracy
   (the rating is ordinal), plus 5-fold cross-validation
        """
    )

    st.warning(
        "**Honest finding:** with 5 balanced rating classes, random guessing scores ~20%. "
        "All three models land within a couple of points of that baseline. None of the "
        "available attributes carry meaningful predictive signal about satisfaction in this "
        "dataset — the `Resolution` text turns out to be randomly-generated placeholder "
        "sentences, not real agent notes. The pipeline itself (leakage-aware feature dropping, "
        "proper encoding, hyperparameter tuning, honestly-tested NLP feature, cross-validated "
        "ordinal-aware evaluation) is complete and reusable on a dataset where the label does "
        "carry signal. Head to **Model Training & Results** to see this for yourself, or "
        "**Predict a Ticket** to try the (weak) model live."
    )

    with st.expander("Preview raw data"):
        st.dataframe(df_raw.head(20), use_container_width=True)

# ============================================================================
# PAGE 2 — Data Explorer
# ============================================================================
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.caption("Interactive EDA on the 2,769 closed tickets that carry a satisfaction rating.")

    with st.sidebar:
        st.markdown("### Filters")
        types = st.multiselect("Ticket Type", sorted(df1["Ticket Type"].unique()),
                                default=list(sorted(df1["Ticket Type"].unique())))
        priorities = st.multiselect("Ticket Priority", sorted(df1["Ticket Priority"].unique()),
                                     default=list(sorted(df1["Ticket Priority"].unique())))
        genders = st.multiselect("Customer Gender", sorted(df1["Customer Gender"].unique()),
                                  default=list(sorted(df1["Customer Gender"].unique())))

    dff = df1[
        df1["Ticket Type"].isin(types)
        & df1["Ticket Priority"].isin(priorities)
        & df1["Customer Gender"].isin(genders)
    ]
    st.caption(f"Showing {len(dff):,} of {len(df1):,} rated tickets after filters.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class balance: Satisfaction Rating")
        rating_counts = dff["Customer Satisfaction Rating"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax, color="#4C72B0")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of tickets")
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Customer age groups")
        chart_age = dff["Type of Customer"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(chart_age, labels=chart_age.index, autopct="%.0f%%",
               colors=sns.color_palette("pastel"))
        st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Ticket type breakdown")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        dff["Ticket Type"].value_counts().plot(kind="barh", ax=ax, color="#55A868")
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)

    with col4:
        st.subheader("Ticket priority breakdown")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        dff["Ticket Priority"].value_counts().plot(kind="barh", ax=ax, color="#C44E52")
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)

    st.subheader("Top 15 products purchased")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dff["Product Purchased"].value_counts().head(15).plot(kind="barh", ax=ax, color="#8172B2")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

    st.subheader("The NLP experiment: resolution sentiment vs. rating")
    corr = dff["Resolution Sentiment"].corr(dff["Customer Satisfaction Rating"])
    nonzero_pct = (dff["Resolution Sentiment"] != 0).mean() * 100
    cA, cB = st.columns(2)
    cA.metric("Correlation with rating", f"{corr:.3f}")
    cB.metric("Notes with any sentiment word", f"{nonzero_pct:.1f}%")
    st.caption(
        "Correlation is essentially zero. Looking at the raw text confirms why: entries are "
        "grammatically-random placeholder sentences, not real support agent notes."
    )
    st.dataframe(dff[["Resolution", "Resolution Sentiment", "Customer Satisfaction Rating"]].sample(
        min(5, len(dff)), random_state=3), use_container_width=True)

# ============================================================================
# PAGE 3 — Model Training & Results
# ============================================================================
elif page == "🤖 Model Training & Results":
    st.title("🤖 Model Training & Results")
    st.caption(
        "Trains Logistic Regression, a tuned Random Forest (GridSearchCV), and an SGD "
        "classifier on an 80/20 stratified split, then checks stability with 5-fold CV."
    )

    with st.spinner("Training models (Random Forest grid search can take a moment the first run)…"):
        bundle = train_all_models(X, y)
        results, preds = compute_results_table(bundle, hash(tuple(bundle["y_test"])))
        cv_results = compute_cv_results(bundle, bundle["rf_best_params"], "cv_v1")

    st.success(f"Random Forest best params: {bundle['rf_best_params']}")

    st.subheader("Single 80/20 split — metrics")
    st.dataframe(results.style.format("{:.3f}"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    results[plot_cols].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    plt.xticks(rotation=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Confusion matrices")
    tabs = st.tabs(list(bundle["models"].keys()))
    labels_sorted = sorted(y.unique())
    for tab, name in zip(tabs, bundle["models"].keys()):
        with tab:
            cm = confusion_matrix(bundle["y_test"], preds[name], labels=labels_sorted)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels_sorted, yticklabels=labels_sorted, ax=ax)
            ax.set_xlabel("Predicted rating")
            ax.set_ylabel("Actual rating")
            st.pyplot(fig, use_container_width=True)

    st.subheader("Random Forest — top 15 feature importances")
    rf_model = bundle["models"]["Random Forest"]["model"]
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(
        ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="#DD8452")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Robustness check: 5-fold cross-validation (on training set)")
    st.dataframe(cv_results.style.format("{:.3f}"), use_container_width=True)
    st.caption(
        "Standard deviations of ~0.01–0.02 across folds confirm the ~20% accuracy is stable — "
        "not a lucky or unlucky single split."
    )

    st.info(
        "**Takeaway:** none of the available customer/ticket/product attributes — including a "
        "genuinely-tested text feature — carry meaningful predictive signal about "
        "`Customer Satisfaction Rating` in this dataset. MAE of ~1.5–1.6 rating points and "
        "'within ±1' accuracy of ~51–54% are close to what you'd get by predicting the middle "
        "rating for everyone. That's a defensible, evidence-backed conclusion, not an "
        "assumption — and the pipeline transfers directly to a dataset where the label does "
        "carry signal."
    )

# ============================================================================
# PAGE 4 — Predict a Ticket
# ============================================================================
elif page == "🎯 Predict a Ticket":
    st.title("🎯 Predict a Ticket's Satisfaction Rating")
    st.caption(
        "Fill in a hypothetical ticket and see what each model guesses. Given the finding on "
        "the Model Training page, treat this as a demo of the pipeline, not a reliable forecast."
    )

    with st.spinner("Preparing models…"):
        bundle = train_all_models(X, y)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Customer Age", 18, 70, 35)
            gender = st.selectbox("Customer Gender", sorted(df1["Customer Gender"].unique()))
            product = st.selectbox("Product Purchased", sorted(product_freq.index))
        with c2:
            ticket_type = st.selectbox("Ticket Type", sorted(df1["Ticket Type"].unique()))
            priority = st.selectbox("Ticket Priority", sorted(df1["Ticket Priority"].unique()))
            channel = st.selectbox("Ticket Channel", sorted(df1["Ticket Channel"].unique()))
        with c3:
            purchase_date = st.date_input("Date of Purchase", value=pd.Timestamp("2023-01-15"))
            responded = st.radio("Has the customer received a first response yet?", ["Yes", "No"])
            model_choice = st.selectbox("Model to use", list(bundle["models"].keys()))

        submitted = st.form_submit_button("Predict satisfaction rating", type="primary")

    if submitted:
        row = {
            "Customer Age": age,
            "Purchase_Month": pd.Timestamp(purchase_date).month,
            "Purchase_DayOfWeek": pd.Timestamp(purchase_date).dayofweek,
            "Resolution Sentiment": 0.0,  # ticket not yet resolved -> no resolution text
            "Product Purchased Freq": product_freq.get(product, 0.0),
        }
        input_df = pd.DataFrame([row])

        cat_row = pd.DataFrame([{
            "Customer Gender": gender,
            "Ticket Type": ticket_type,
            "Ticket Priority": priority,
            "Ticket Channel": channel,
            "Type of Customer": bucket_age(age),
            "Response": responded,
        }])
        cat_dummies = pd.get_dummies(cat_row, columns=ONEHOT_COLS, drop_first=False)

        full_row = pd.concat([input_df, cat_dummies], axis=1)
        full_row = full_row.reindex(columns=X.columns, fill_value=0)

        info = bundle["models"][model_choice]
        if info["scaled"]:
            X_input = bundle["scaler"].transform(full_row)
        else:
            X_input = full_row

        pred = info["model"].predict(X_input)[0]

        st.markdown("### Result")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Predicted satisfaction rating", f"{pred} / 5")
        with r2:
            if hasattr(info["model"], "predict_proba"):
                proba = info["model"].predict_proba(X_input)[0]
                classes = info["model"].classes_
                fig, ax = plt.subplots(figsize=(5, 2.5))
                sns.barplot(x=[str(c) for c in classes], y=proba, ax=ax, color="#4C72B0")
                ax.set_xlabel("Rating")
                ax.set_ylabel("Predicted probability")
                st.pyplot(fig, use_container_width=True)
            else:
                st.caption("This model doesn't expose class probabilities.")

        st.warning(
            "Reminder: on held-out data this model is only ~2 points above random-guess "
            "accuracy (~20% for 5 classes). Don't read much into any single prediction above — "
            "see **Model Training & Results** for the full evidence."
        )
