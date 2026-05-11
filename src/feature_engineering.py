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
    """
    df = df.copy()

    df["EngagementScore"] = df["likes_received"] + df["message_sent_count"]
    print("[FeatureEng] Created EngagementScore = likes_received + message_sent_count")

    df["ProfileQuality"] = df["bio_length"] + 5 * df["profile_pics_count"]
    print("[FeatureEng] Created ProfileQuality  = bio_length + 5 * profile_pics_count")

    df["ActivityIntensity"] = df["app_usage_time_min"] * df["swipe_right_ratio"]
    print("[FeatureEng] Created ActivityIntensity = app_usage_time_min x swipe_right_ratio")

    return df
