"""
feature_engineering.py
-----------------------
Create three engineered features that capture composite behavioral signals.

These are added to the DataFrame BEFORE the preprocessing pipeline runs,
so the ColumnTransformer picks them up as regular numerical features.
"""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the DataFrame (in-place copy).

    Features
    --------
    EngagementScore   = likes_received + message_sent_count
        → Combined passive (likes) + active (messages) interaction behaviour.

    ProfileQuality    = bio_length + 5 * profile_pics_count
        → Profile effort metric; pictures are weighted higher because they
          have outsized impact on first impressions.

    ActivityIntensity = app_usage_time_min × swipe_right_ratio
        → Captures combined platform engagement and swipe aggressiveness.

    interest_tags_count = Number of interests listed in interest_tags
        → More interests may indicate a more complete profile.

    EngagementIntensity = emoji_usage_rate × message_sent_count
        → High emoji usage combined with high message count signals high enthusiasm.
    """
    df = df.copy()

    df["EngagementScore"] = df["likes_received"] + df["message_sent_count"]
    print("[FeatureEng] Created EngagementScore = likes_received + message_sent_count")

    df["ProfileQuality"] = df["bio_length"] + 5 * df["profile_pics_count"]
    print("[FeatureEng] Created ProfileQuality  = bio_length + 5 * profile_pics_count")

    df["ActivityIntensity"] = df["app_usage_time_min"] * df["swipe_right_ratio"]
    print("[FeatureEng] Created ActivityIntensity = app_usage_time_min x swipe_right_ratio")

    # New Features
    if "interest_tags" in df.columns:
        # Count tags by counting commas + 1. If na, 0.
        df["interest_tags_count"] = df["interest_tags"].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        print("[FeatureEng] Created interest_tags_count")

    df["EngagementIntensity"] = df["emoji_usage_rate"] * df["message_sent_count"]
    print("[FeatureEng] Created EngagementIntensity = emoji_usage_rate x message_sent_count")

    return df
