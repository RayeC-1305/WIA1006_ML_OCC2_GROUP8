"""
preprocessing.py
-----------------
Data loading, cleaning, target grouping, and sklearn pipeline construction.

Pipeline architecture (industry-standard):
    Numerical  → SimpleImputer(median) → StandardScaler
    Categorical → SimpleImputer(most_frequent) → OneHotEncoder
Combined via ColumnTransformer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Global label encoder – fitted during load_and_clean()
label_encoder = LabelEncoder()


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
# Columns to DROP (leakage / redundancy / multi-value text)
DROP_COLS = [
    "mutual_matches",       # Target leakage – directly encodes match info
    "app_usage_time_label", # Derived bin of app_usage_time_min (redundant)
    "swipe_right_label",    # Derived bin of swipe_right_ratio (redundant)
    "interest_tags",        # Multi-value text – out of scope
    "sexual_orientation",   # Not in recommended feature list
]

# Final feature lists (AFTER feature engineering adds 3 extra numerical cols)
NUMERICAL_FEATURES = [
    "app_usage_time_min",
    "swipe_right_ratio",
    "likes_received",
    "profile_pics_count",
    "bio_length",
    "message_sent_count",
    "emoji_usage_rate",
    "last_active_hour",
    # Engineered features (added by feature_engineering.py)
    "EngagementScore",
    "ProfileQuality",
    "ActivityIntensity",
]

CATEGORICAL_FEATURES = [
    "gender",
    "education_level",
    "income_bracket",
    "location_type",
    "swipe_time_of_day",
]

TARGET = "match_outcome"

# ---------------------------------------------------------------------------
# Target grouping  (Option B – 3 categories)
# ---------------------------------------------------------------------------
TARGET_MAP = {
    # Positive outcomes
    "Mutual Match":        "Positive",
    "Date Happened":       "Positive",
    "Relationship Formed": "Positive",
    "Instant Match":       "Positive",
    # Neutral outcomes
    "No Action":           "Neutral",
    "One-sided Like":      "Neutral",
    "Chat Ignored":        "Neutral",
    # Negative outcomes
    "Ghosted":             "Negative",
    "Blocked":             "Negative",
    "Catfished":           "Negative",
}


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and perform initial cleaning."""
    df = pd.read_csv(csv_path)
    print(f"[Preprocessing] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} cols")

    # Drop leakage / redundant / text columns
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"[Preprocessing] Dropped columns: {existing_drops}")

    # Map target to 3-class grouping
    df[TARGET] = df[TARGET].map(TARGET_MAP)
    unmapped = df[TARGET].isna().sum()
    if unmapped:
        print(f"[Preprocessing] WARNING: {unmapped} rows with unmapped target - dropping")
        df = df.dropna(subset=[TARGET])

    print(f"[Preprocessing] Target distribution after grouping:")
    print(df[TARGET].value_counts())

    # Encode target as integers (required by XGBoost)
    df[TARGET] = label_encoder.fit_transform(df[TARGET])
    global CLASS_NAMES
    CLASS_NAMES = list(label_encoder.classes_)
    print(f"[Preprocessing] Encoded target classes: {CLASS_NAMES}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer with numerical + categorical pipelines."""
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train-test split."""
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Preprocessing] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
