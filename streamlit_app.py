"""
streamlit_app.py — "Tying the (Data) Knot" Interactive Dashboard
Run: streamlit run streamlit_app.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is on the path for src imports
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.feature_engineering import add_engineered_features
from src.preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    TARGET_MAP,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(ROOT, "data", "dating_app_behavior_dataset.csv")
MODELS_DIR = os.path.join(ROOT, "models")
VIZ_DIR = os.path.join(ROOT, "visualizations")
REPORTS_DIR = os.path.join(ROOT, "reports")

CLASS_NAMES = ["Negative", "Neutral", "Positive"]

OUTCOME_INFO = {
    "Positive": {
        "color": "#34C759",
        "icon": "+",
        "includes": ["Mutual Match", "Date Happened", "Relationship Formed", "Instant Match"],
    },
    "Neutral": {
        "color": "#FFCC00",
        "icon": "~",
        "includes": ["No Action", "One-sided Like", "Chat Ignored"],
    },
    "Negative": {
        "color": "#FF3B30",
        "icon": "-",
        "includes": ["Ghosted", "Blocked", "Catfished"],
    },
}

# ── Caching ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(name: str = "xgboost"):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_comparison():
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


# Models trained with the standard feature set (compatible with current preprocessing)
COMPATIBLE_MODELS = [
    "xgboost", "random_forest", "decision_tree", "logistic_regression",
    "neural_network_mlp", "svm", "voting_classifier", "dummy_baseline",
    "xgboost_smote", "random_forest_smote", "decision_tree_smote",
    "logistic_regression_smote", "neural_network_mlp_smote", "svm_smote",
    "dummy_baseline_smote",
]


def list_model_files():
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(
        f.replace(".pkl", "")
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl") and f.replace(".pkl", "") in COMPATIBLE_MODELS
    )


def list_viz_files():
    if not os.path.isdir(VIZ_DIR):
        return []
    return sorted(f for f in os.listdir(VIZ_DIR) if f.endswith(".png"))


# ── Prediction helper ────────────────────────────────────────────────────────

def predict_match(model, input_dict: dict):
    df = pd.DataFrame([input_dict])
    df = add_engineered_features(df)
    pred_idx = model.predict(df)[0]
    probs = model.predict_proba(df)[0]
    outcome = CLASS_NAMES[int(pred_idx)]
    return outcome, probs


def render_result(outcome, probs):
    info = OUTCOME_INFO[outcome]
    st.markdown(
        f"<h1 style='text-align:center; color:{info['color']}; font-size:3rem; margin-bottom:0;'>"
        f"{outcome.upper()}</h1>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    for col, cls, prob in zip(
        [col1, col2, col3], CLASS_NAMES, probs
    ):
        c = OUTCOME_INFO[cls]["color"]
        with col:
            st.markdown(
                f"<div style='text-align:center; padding:1rem; border-radius:12px; "
                f"background:{c}22; border:1px solid {c};'>"
                f"<span style='font-size:1.5rem; font-weight:700; color:{c};'>{prob*100:.1f}%</span><br>"
                f"<span style='color:{c}; font-size:0.9rem;'>{cls}</span></div>",
                unsafe_allow_html=True,
            )

    # Probability bar chart
    fig = px.bar(
        x=CLASS_NAMES,
        y=[p * 100 for p in probs],
        color=CLASS_NAMES,
        color_discrete_map={c: OUTCOME_INFO[c]["color"] for c in CLASS_NAMES},
        labels={"x": "Outcome", "y": "Probability (%)"},
        title="Prediction Confidence",
    )
    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # What each category includes
    with st.expander("What does each outcome include? (from original dataset)"):
        for cls in CLASS_NAMES:
            info = OUTCOME_INFO[cls]
            tags = ", ".join(info["includes"])
            st.markdown(f"**{cls}:** {tags}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## Tying the (Data) Knot")
        st.markdown("*Love, Life & Likes*")
        st.divider()
        st.markdown(
            "**WIA1006/WID3006 Machine Learning**\n\n"
            "Predicting dating app match outcomes from "
            "50,000 synthetic user profiles using XGBoost, "
            "Random Forest, and more."
        )
        st.divider()
        st.markdown("**Group 8 Members:**")
        members = [
            "Raye Chan Jun Foong (Leader)",
            "Yip Zheng Xyun",
            "Hon Chi Fung",
            "Ong Zheng Xi",
            "Daniel Goh Zhi Qian",
            "Lau Wei Zhong",
        ]
        for m in members:
            st.markdown(f"- {m}")


# ── Tab 1: Match Survey ─────────────────────────────────────────────────────

def tab_survey():
    st.markdown("## Dating Profile Survey")
    st.markdown(
        "Answer a few questions about your dating habits and we'll predict your match outcome."
    )

    with st.form("survey_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.radio("What's your gender?", ["Male", "Female", "Non-binary", "Other"])
            age = st.slider("Your age", 18, 65, 25)
            location = st.selectbox("Where do you live?", ["Urban", "Suburban", "Rural"])
            income = st.selectbox("Income bracket", ["Low", "Middle", "High"])
            education = st.selectbox("Education level", ["High School", "Bachelor's", "Master's", "PhD"])

        with col2:
            usage = st.slider("Daily app usage (minutes)", 0, 300, 45)
            swipe_style = st.select_slider(
                "Swipe style",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate",
            )
            profile_effort = st.select_slider(
                "How much effort do you put into your profile?",
                options=["Minimal", "Average", "High effort"],
                value="Average",
            )
            active_time = st.selectbox(
                "When are you most active?",
                ["Morning", "Afternoon", "Evening", "Night"],
            )

        submitted = st.form_submit_button("Predict My Match Outcome", use_container_width=True)

    if submitted:
        # Map survey answers to model features
        swipe_map = {"Conservative": 0.25, "Moderate": 0.50, "Aggressive": 0.80}
        effort_map = {
            "Minimal": {"profile_pics_count": 1, "bio_length": 50},
            "Average": {"profile_pics_count": 3, "bio_length": 200},
            "High effort": {"profile_pics_count": 6, "bio_length": 400},
        }
        time_map = {"Morning": 9, "Afternoon": 14, "Evening": 20, "Night": 23}

        eff = effort_map[profile_effort]
        input_dict = {
            "gender": gender,
            "age": age,
            "location_type": location,
            "app_usage_time_min": usage,
            "swipe_right_ratio": swipe_map[swipe_style],
            "swipe_time_of_day": active_time,
            "last_active_hour": time_map[active_time],
            "profile_pics_count": eff["profile_pics_count"],
            "bio_length": eff["bio_length"],
            "interest_tags": "Music, Travel, Coffee",
            "likes_received": max(10, usage),
            "message_sent_count": max(5, usage // 2),
            "emoji_usage_rate": 0.25 if profile_effort == "Minimal" else 0.35,
            "income_bracket": income,
            "education_level": education,
        }

        model = load_model("xgboost")
        if model is None:
            st.error("Could not load models/xgboost.pkl. Run main.py first to train models.")
            return

        outcome, probs = predict_match(model, input_dict)
        st.divider()
        render_result(outcome, probs)


# ── Tab 2: Custom Prediction ────────────────────────────────────────────────

def tab_custom():
    st.markdown("## Custom Prediction")
    st.markdown("Fine-tune every feature and see how it affects the match outcome.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Demographics")
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"], key="c_gender")
        age = st.number_input("Age", 18, 100, 25, key="c_age")
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"], key="c_loc")
        income = st.selectbox("Income Bracket", ["Low", "Middle", "High"], key="c_inc")
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"], key="c_edu")

        st.markdown("### App Behavior")
        usage = st.slider("Daily App Usage (min)", 0, 300, 45, key="c_usage")
        swipe = st.slider("Swipe Right Ratio (%)", 0, 100, 50, key="c_swipe")
        active_time = st.selectbox("Swipe Time", ["Morning", "Afternoon", "Evening", "Night"], key="c_time")

    with col2:
        st.markdown("### Profile")
        pics = st.slider("Profile Pictures", 0, 10, 4, key="c_pics")
        bio = st.slider("Bio Length (chars)", 0, 500, 150, key="c_bio")
        interests = st.text_input("Interests (comma separated)", "Music, Travel, Coffee", key="c_int")
        likes = st.slider("Likes Received", 0, 500, 100, key="c_likes")
        messages = st.slider("Messages Sent", 0, 300, 50, key="c_msg")
        emoji = st.slider("Emoji Usage Rate (%)", 0, 100, 20, key="c_emoji")
        last_hour = st.slider("Last Active Hour (0-23)", 0, 23, 20, key="c_hour")

    if st.button("Predict Outcome", use_container_width=True, key="c_predict"):
        time_map = {"Morning": "Morning", "Afternoon": "Afternoon", "Evening": "Evening", "Night": "Night"}
        input_dict = {
            "gender": gender,
            "age": age,
            "location_type": location,
            "app_usage_time_min": usage,
            "swipe_right_ratio": swipe / 100.0,
            "swipe_time_of_day": active_time,
            "last_active_hour": last_hour,
            "profile_pics_count": pics,
            "bio_length": bio,
            "interest_tags": interests,
            "likes_received": likes,
            "message_sent_count": messages,
            "emoji_usage_rate": emoji / 100.0,
            "income_bracket": income,
            "education_level": education,
        }

        model = load_model("xgboost")
        if model is None:
            st.error("Could not load models/xgboost.pkl. Run main.py first.")
            return

        outcome, probs = predict_match(model, input_dict)
        st.divider()
        render_result(outcome, probs)


# ── Tab 3: Model Comparison ─────────────────────────────────────────────────

def tab_models():
    st.markdown("## Model Comparison & Results")
    st.markdown("Evaluate all trained models and explore their performance visually.")

    comparison = load_comparison()
    if comparison is not None:
        st.markdown("### Performance Rankings")
        st.dataframe(
            comparison.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1_Weighted": "{:.4f}",
            }),
            use_container_width=True,
        )

    st.divider()
    st.markdown("### Visualizations")

    viz_files = list_viz_files()
    viz_labels = {
        "05_confusion_matrices.png": "Confusion Matrices",
        "06_roc_curves.png": "ROC Curves (One-vs-Rest)",
        "07_model_comparison.png": "Model Comparison Bar Chart",
        "08_feature_importance_random_forest.png": "Feature Importance — Random Forest",
        "08_feature_importance_xgboost.png": "Feature Importance — XGBoost",
        "09_learning_curves.png": "Learning Curves (Overfitting Analysis)",
        "10_shap_summary_xgboost.png": "SHAP Summary — XGBoost",
        "10_shap_summary_random_forest.png": "SHAP Summary — Random Forest",
    }

    available = [f for f in viz_files if f in viz_labels]
    if available:
        selected = st.selectbox(
            "Select visualization",
            available,
            format_func=lambda f: viz_labels.get(f, f),
        )
        st.image(os.path.join(VIZ_DIR, selected), use_container_width=True)
    else:
        st.info("No visualizations found. Run main.py first to generate them.")

    st.divider()
    st.markdown("### Live Model Evaluation")

    model_names = [n for n in list_model_files() if "dummy" not in n]
    if model_names:
        chosen = st.selectbox("Select a model to evaluate", model_names)
        if st.button("Run Evaluation", key="eval_btn"):
            with st.spinner(f"Loading {chosen} and running predictions..."):
                model = load_model(chosen)
                if model is None:
                    st.error(f"Could not load {chosen}.pkl")
                    return

                df = load_data()
                df_clean = df.copy()

                # Replicate preprocessing
                drop_cols = ["mutual_matches", "app_usage_time_label", "swipe_right_label", "sexual_orientation"]
                existing = [c for c in drop_cols if c in df_clean.columns]
                df_clean = df_clean.drop(columns=existing)
                df_clean[TARGET] = df_clean[TARGET].map(TARGET_MAP)
                df_clean = df_clean.dropna(subset=[TARGET])
                df_clean = add_engineered_features(df_clean)

                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, f1_score, classification_report

                feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
                X = df_clean[feature_cols]
                y_raw = df_clean[TARGET]

                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y_raw)

                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{acc:.4f}")
                m2.metric("F1 (Weighted)", f"{f1:.4f}")

                st.text(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))


# ── Tab 4: Data Explorer ────────────────────────────────────────────────────

def tab_explorer():
    st.markdown("## Data Explorer")
    st.markdown("Interactive exploration of the dating app dataset — equivalent to the Jupyter notebook.")

    df = load_data()

    # Dataset overview
    st.markdown("### Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Match Outcomes", df["match_outcome"].nunique())

    with st.expander("View first rows"):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Column data types"):
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}), use_container_width=True)

    # Summary statistics
    st.markdown("### Summary Statistics")
    tab_num, tab_cat = st.tabs(["Numerical", "Categorical"])

    with tab_num:
        st.dataframe(df.describe(), use_container_width=True)
    with tab_cat:
        st.dataframe(df.describe(include=["object"]), use_container_width=True)

    st.divider()

    # Target distribution
    st.markdown("### Match Outcome Distribution")
    target_mapped = df["match_outcome"].map(TARGET_MAP)
    counts = target_mapped.value_counts()

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map={c: OUTCOME_INFO[c]["color"] for c in CLASS_NAMES},
        labels={"x": "Match Outcome", "y": "Count"},
        title="Grouped Target Distribution (3 Classes)",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Original 10-class distribution
    orig_counts = df["match_outcome"].value_counts()
    fig2 = px.bar(
        x=orig_counts.index,
        y=orig_counts.values,
        labels={"x": "Match Outcome", "y": "Count"},
        title="Original 10-Class Distribution",
        color=orig_counts.values,
        color_continuous_scale="viridis",
    )
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Pearson Correlation Heatmap")
        st.pyplot(fig_corr)

    st.divider()

    # Feature distributions by target
    st.markdown("### Feature Distributions by Match Outcome")
    numeric_options = [c for c in num_cols if c != TARGET]
    selected_feat = st.multiselect(
        "Select features to compare",
        numeric_options,
        default=numeric_options[:4],
    )
    if selected_feat:
        df_plot = df.copy()
        df_plot["outcome_group"] = df_plot["match_outcome"].map(TARGET_MAP)
        for feat in selected_feat:
            fig_dist = px.histogram(
                df_plot,
                x=feat,
                color="outcome_group",
                color_discrete_map={c: OUTCOME_INFO[c]["color"] for c in CLASS_NAMES},
                barmode="overlay",
                opacity=0.6,
                title=f"{feat} by Match Outcome",
            )
            fig_dist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # Feature engineering showcase
    st.markdown("### Engineered Features")
    st.markdown("We created 5 composite features to capture behavioral signals:")
    features_info = [
        ("EngagementScore", "likes_received + message_sent_count", "Passive + active interaction"),
        ("ProfileQuality", "bio_length + 5 * profile_pics_count", "Profile effort metric"),
        ("ActivityIntensity", "app_usage_time_min * swipe_right_ratio", "Engagement x aggressiveness"),
        ("interest_tags_count", "count(interest_tags.split(','))", "Profile completeness"),
        ("EngagementIntensity", "emoji_usage_rate * message_sent_count", "Enthusiasm signal"),
    ]
    feat_df = pd.DataFrame(features_info, columns=["Feature", "Formula", "Description"])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ── Tab 5: Project Report ───────────────────────────────────────────────────

def tab_report():
    st.markdown("## Project Report")
    report_path = os.path.join(ROOT, "Group_Project_Report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.error("Group_Project_Report.md not found.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Tying the (Data) Knot",
        page_icon=":heart:",
        layout="wide",
    )

    render_sidebar()

    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>Tying the (Data) Knot</h1>"
        "<p style='text-align:center; color:gray; margin-top:0;'>Dating App Match Prediction Dashboard</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Match Survey",
        "Custom Prediction",
        "Model Comparison",
        "Data Explorer",
        "Project Report",
    ])

    with tab1:
        tab_survey()
    with tab2:
        tab_custom()
    with tab3:
        tab_models()
    with tab4:
        tab_explorer()
    with tab5:
        tab_report()


if __name__ == "__main__":
    main()
