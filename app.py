import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

st.set_page_config(
    page_title="AI Fraud Detection & Risk Intelligence",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "best_fraud_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🛡️ AI Fraud Detection & Risk Intelligence System")
st.markdown(
    "### Intelligent transaction screening using Machine Learning"
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file missing. Place best_fraud_model.pkl beside app.py.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

features = getattr(model, "feature_names_in_", None)

if features is None:
    st.error(
        "The model does not expose feature names. "
        "We must restore the original training feature names before prediction."
    )
    st.stop()

features = list(features)

st.sidebar.header("⚙️ Configuration")
threshold = st.sidebar.slider(
    "Fraud risk threshold (%)", 1, 99, 50
) / 100

st.sidebar.markdown("---")
st.sidebar.write("Model: XGBoost")
st.sidebar.write(f"Required features: {len(features)}")

st.subheader("1. Upload transaction data")
uploaded = st.file_uploader(
    "Upload your testing CSV or a new transaction CSV",
    type=["csv"]
)

if uploaded is None:
    st.info("Upload your existing 20% testing CSV to begin.")
    st.write("**Dashboard features**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "XGBoost")
    c2.metric("Input Features", len(features))
    c3.metric("Status", "Ready")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"CSV reading failed: {e}")
    st.stop()

if data.empty:
    st.error("The uploaded file is empty.")
    st.stop()

st.write("Preview of uploaded transactions")
st.dataframe(data.head(10), use_container_width=True)

missing = [c for c in features if c not in data.columns]

if missing:
    st.error("Required input columns are missing:")
    st.write(missing)
    st.write("Ensure the CSV uses the same input columns as model training.")
    st.stop()

# Optional actual target label for evaluation only
possible_targets = [
    "isFraud", "is_fraud", "fraud", "Class",
    "class", "label", "target", "isFraud?"
]
target_candidates = [
    c for c in data.columns
    if c not in features and c in possible_targets
]

target_col = None
if target_candidates:
    target_col = st.selectbox(
        "Actual fraud-label column (optional; used only for evaluation)",
        ["None"] + target_candidates
    )
    if target_col == "None":
        target_col = None

X = data[features].copy()

try:
    # Handle missing or infinite numeric values only where supported
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)

    pred = model.predict(X)

    if not hasattr(model, "predict_proba"):
        st.error("This model cannot provide probability-based risk scores.")
        st.stop()

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 not in classes:
        st.error(
            "Class 1 was not found. Confirm the fraud label encoding "
            "before interpreting risk scores."
        )
        st.stop()

    fraud_probability = probabilities[:, classes.index(1)]

except Exception as e:
    st.error(f"Prediction failed. Check data types and preprocessing: {e}")
    st.stop()

results = data.copy()
results["Fraud_Risk_Score_Percent"] = np.round(
    fraud_probability * 100, 2
)
results["Predicted_Label"] = pred
results["Risk_Decision"] = np.where(
    fraud_probability >= threshold,
    "Potential Fraud",
    "Likely Legitimate"
)

fraud_mask = fraud_probability >= threshold
fraud_count = int(fraud_mask.sum())
total = len(results)

st.subheader("2. Fraud risk overview")
a, b, c, d = st.columns(4)
a.metric("Transactions Analyzed", f"{total:,}")
b.metric("Potential Fraud Flags", f"{fraud_count:,}")
c.metric("Likely Legitimate", f"{total - fraud_count:,}")
d.metric("Flagged Rate", f"{fraud_count / total * 100:.2f}%")

st.caption(
    "Risk flags are model predictions, not proof of fraud. "
    "The threshold changes displayed decisions, not the trained model."
)

left, right = st.columns(2)

with left:
    st.markdown("**Transaction decision distribution**")
    st.bar_chart(results["Risk_Decision"].value_counts())

with right:
    st.markdown("**Fraud risk score distribution**")
    fig, ax = plt.subplots()
    ax.hist(fraud_probability * 100, bins=20)
    ax.set_xlabel("Fraud risk score (%)")
    ax.set_ylabel("Transactions")
    ax.set_title("Risk Score Distribution")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("3. Filter and inspect transactions")
filter_choice = st.selectbox(
    "Choose records",
    ["All", "Potential Fraud", "Likely Legitimate"]
)

view = results
if filter_choice != "All":
    view = results[results["Risk_Decision"] == filter_choice]

st.dataframe(view, use_container_width=True)

st.download_button(
    "⬇️ Download prediction results",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="fraud_detection_predictions.csv",
    mime="text/csv"
)

# Evaluation uses actual labels only if they are supplied
if target_col is not None:
    st.subheader("4. Model evaluation on labelled data")
    y_true = data[target_col]

    try:
        y_true_num = pd.to_numeric(y_true)
        if not set(y_true_num.dropna().unique()).issubset({0, 1}):
            st.warning(
                "The selected target must use 0 for legitimate "
                "and 1 for fraud for these metrics."
            )
        else:
            y_true_num = y_true_num.astype(int)
            y_hat = (fraud_probability >= threshold).astype(int)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_true_num, y_hat):.4f}")
            m2.metric("Precision", f"{precision_score(y_true_num, y_hat, zero_division=0):.4f}")
            m3.metric("Recall", f"{recall_score(y_true_num, y_hat, zero_division=0):.4f}")
            m4.metric("F1 Score", f"{f1_score(y_true_num, y_hat, zero_division=0):.4f}")

            cm = confusion_matrix(y_true_num, y_hat, labels=[0, 1])
            st.write("Confusion matrix (actual rows, predicted columns)")
            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual Legitimate", "Actual Fraud"],
                    columns=["Predicted Legitimate", "Predicted Fraud"]
                )
            )

            if len(np.unique(y_true_num)) == 2:
                auc = roc_auc_score(y_true_num, fraud_probability)
                st.metric("ROC-AUC", f"{auc:.4f}")

                fpr, tpr, _ = roc_curve(y_true_num, fraud_probability)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                st.text(
                    classification_report(
                        y_true_num, y_hat, zero_division=0
                    )
                )
    except Exception as e:
        st.warning(f"Could not calculate evaluation metrics: {e}")

st.success("Transaction analysis completed.")
