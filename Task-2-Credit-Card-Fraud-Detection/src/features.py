import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["cc_num", "merchant", "category", "gender", "state", "job", "city", "zip"]
FEATURE_COLUMNS = [
    "amt", "city_pop", "lat", "long", "merch_lat", "merch_long",
    "hour", "dayofweek", "month", "day", "age", "distance_km",
    *CATEGORICAL_COLUMNS,
]


def make_features(df, categorical_levels=None):
    x = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    dob = pd.to_datetime(df["dob"], errors="coerce")

    for c in ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long"]:
        x[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    x["hour"] = dt.dt.hour.fillna(0).astype("int8")
    x["dayofweek"] = dt.dt.dayofweek.fillna(0).astype("int8")
    x["month"] = dt.dt.month.fillna(0).astype("int8")
    x["day"] = dt.dt.day.fillna(0).astype("int8")
    x["age"] = (dt.dt.year - dob.dt.year).fillna(0).astype("int16")

    lat1 = np.radians(x["lat"].to_numpy())
    lat2 = np.radians(x["merch_lat"].to_numpy())
    dlat = lat2 - lat1
    dlon = np.radians(x["merch_long"].to_numpy() - x["long"].to_numpy())
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    x["distance_km"] = (6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype("float32")

    for c in CATEGORICAL_COLUMNS:
        if categorical_levels and c in categorical_levels:
            x[c] = pd.Categorical(df[c].astype(str), categories=categorical_levels[c])
        else:
            x[c] = df[c].astype("category")

    return x[FEATURE_COLUMNS]
