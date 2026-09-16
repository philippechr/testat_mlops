"""Prepare features and training rows from real observations only.

No interpolation, generated measurements or forward filling is performed.
All timestamps entering/leaving this module are Unix milliseconds in UTC.
"""

import pandas as pd

HOUR_MS = 3_600_000
WINDOW_MS = 3 * HOUR_MS
MIN_HOURLY_BINS = 3
MAX_GAP_MS = 2 * HOUR_MS
HORIZON_MS = 3 * HOUR_MS
LABEL_TOLERANCE_MS = HOUR_MS // 2

# The training and inference scripts must use this same ordered list.
MODEL_FEATURES = ["temperature_mean_3h", "temperature_current"]

FEATURE_TYPES = {
    "location_id": "object",
    "observed_at": "int64",
    "temperature_mean_3h": "float64",
    "temperature_current": "float64",
    "history_hour_count": "int64",
    "history_observation_count": "int64",
    "history_start_at": "int64",
    "history_end_at": "int64",
}
TRAINING_TYPES = {
    **FEATURE_TYPES,
    "temperature_target_3h": "float64",
    "target_observed_at": "int64",
    "target_offset_minutes": "float64",
}


def typed_frame(rows, types):
    return pd.DataFrame(rows, columns=list(types)).astype(types)


def feature_for_time(history, location_id, observed_at, temperature_current):
    """Reusable calculation for historical training and later live inference.

    history must contain observations for this location only. Current RT
    temperature is supplied separately and never enters the past-3h mean.
    A None result means insufficient real history; no fallback is invented.
    """
    observed_at = int(observed_at)
    start = observed_at - WINDOW_MS
    history = history.loc[history["location_id"] == location_id].sort_values(
        "observed_at"
    )
    if history.empty or int(history["observed_at"].min()) > start:
        return None
    past = history.loc[
        (history["observed_at"] >= start) & (history["observed_at"] < observed_at)
    ].copy()
    if past.empty:
        return None

    # Slots are anchored to the beginning of the exact 3h window.
    # Extra calls in a slot contribute to that slot's mean, not extra weight.
    past["hour_slot"] = (past["observed_at"] - start) // HOUR_MS
    hourly = past.groupby("hour_slot")["temperature_c"].mean()
    if len(hourly) < MIN_HOURLY_BINS:
        return None

    # Include the edges of the window in the gap check.
    times = [start, *past["observed_at"].tolist(), observed_at]
    if max(right - left for left, right in zip(times, times[1:])) > MAX_GAP_MS:
        return None

    return {
        "location_id": str(location_id),
        "observed_at": observed_at,
        "temperature_mean_3h": float(hourly.mean()),
        "temperature_current": float(temperature_current),
        "history_hour_count": int(len(hourly)),
        "history_observation_count": int(len(past)),
        "history_start_at": int(past["observed_at"].min()),
        "history_end_at": int(past["observed_at"].max()),
    }


def prepare_features(observations):
    """Return inference-capable feature rows and rows with mature real labels."""
    feature_rows, training_rows = [], []
    for location_id, group in observations.groupby("location_id", sort=True):
        group = group.sort_values("observed_at").drop_duplicates(
            "observed_at", keep="last"
        )
        latest_observation = int(group["observed_at"].max())
        for row in group.itertuples(index=False):
            features = feature_for_time(
                group, location_id, row.observed_at, row.temperature_c
            )
            if features is None:
                continue
            feature_rows.append(features)

            target_time = int(row.observed_at) + HORIZON_MS
            # Wait for the whole matching window to pass. This avoids selecting
            # an early label before a closer subsequent observation arrives.
            if latest_observation < target_time + LABEL_TOLERANCE_MS:
                continue
            candidates = group.loc[
                group["observed_at"].between(
                    target_time - LABEL_TOLERANCE_MS,
                    target_time + LABEL_TOLERANCE_MS,
                )
            ].copy()
            if candidates.empty:
                continue
            candidates["distance"] = (candidates["observed_at"] - target_time).abs()
            target = candidates.sort_values(["distance", "observed_at"]).iloc[0]
            training_rows.append(
                {
                    **features,
                    "temperature_target_3h": float(target["temperature_c"]),
                    "target_observed_at": int(target["observed_at"]),
                    "target_offset_minutes": float(
                        (int(target["observed_at"]) - target_time) / 60_000
                    ),
                }
            )
    return typed_frame(feature_rows, FEATURE_TYPES), typed_frame(
        training_rows, TRAINING_TYPES
    )
