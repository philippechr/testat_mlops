"""Train on real Hopsworks data and track one run in MLflow."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import hopsworks
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from hopsworks.client.exceptions import RestAPIError
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from weather_features import MODEL_FEATURES, HORIZON_MS, LABEL_TOLERANCE_MS

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
TARGET = "temperature_target_3h"
GROUP_NAME, GROUP_VERSION = "weather_training_features", 2
VIEW_NAME, VIEW_VERSION = "weather_temperature_training", 2
COLUMNS = ["location_id", "observed_at", *MODEL_FEATURES, TARGET, "target_observed_at"]
MIN_ROWS, MIN_TRAIN_ROWS, MIN_TEST_ROWS = 24, 12, 5


class NotReady(Exception):
    pass


def required(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Fehlender Eintrag: {name}")
    return value


def epoch_ms(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError("Fehlender oder ungültiger Zeitstempel.")
        return values.astype("int64")
    dates = pd.to_datetime(series, utc=True, errors="raise")
    if dates.isna().any():
        raise ValueError("Fehlender Zeitstempel.")
    return dates.map(lambda value: value.value // 1_000_000).astype("int64")


def normalize(frame, location_id=None):
    if frame.empty:
        raise NotReady("Die Trainings-Feature-Group enthält noch keine Zeilen.")
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {sorted(missing)}")
    frame = frame[COLUMNS].copy()
    if frame.isna().any().any():
        raise ValueError(
            "Trainingsdaten enthalten leere Werte. Feature-Pipeline prüfen."
        )
    frame["location_id"] = frame["location_id"].astype(str)
    if location_id:
        frame = frame.loc[frame["location_id"] == location_id].copy()
    if frame.empty:
        raise NotReady("Keine Trainingszeilen für den gewählten Standort vorhanden.")
    if frame["location_id"].nunique() != 1:
        raise ValueError(
            "Mehrere Standorte vorhanden. Einen mit --location-id auswählen."
        )
    for name in ["observed_at", "target_observed_at"]:
        frame[name] = epoch_ms(frame[name])
    for name in [*MODEL_FEATURES, TARGET]:
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype("float64")
    if not np.isfinite(frame[[*MODEL_FEATURES, TARGET]].to_numpy()).all():
        raise ValueError("Trainingsdaten enthalten nicht-endliche Temperaturen.")
    if frame.duplicated(["location_id", "observed_at"]).any():
        raise ValueError("Doppelte Trainingsschlüssel gefunden. Feature Group prüfen.")
    offsets = frame["target_observed_at"] - frame["observed_at"] - HORIZON_MS
    if (offsets.abs() > LABEL_TOLERANCE_MS).any():
        raise ValueError(
            "Mindestens ein Label liegt ausserhalb der erlaubten Zeittoleranz."
        )
    return frame.sort_values("observed_at").reset_index(drop=True)


def split_data(frame):
    if len(frame) < MIN_ROWS:
        raise NotReady(
            f"Erst {len(frame)} vollständige Trainingszeilen; mindestens {MIN_ROWS} benötigt."
        )
    cutoff_index = int(len(frame) * 0.8)
    cutoff = int(frame.iloc[cutoff_index]["observed_at"])
    older = frame.loc[frame["observed_at"] < cutoff]
    # The whole label matching window must be in the past at test start.
    train = older.loc[
        (older["observed_at"] + HORIZON_MS + LABEL_TOLERANCE_MS < cutoff)
        & (older["target_observed_at"] < cutoff)
    ].copy()
    test = frame.loc[frame["observed_at"] >= cutoff].copy()
    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
        raise NotReady(
            f"Nach zeitlicher Trennung: {len(train)} Training, {len(test)} Test. "
            f"Benötigt: mindestens {MIN_TRAIN_ROWS} bzw. {MIN_TEST_ROWS}. Weiter sammeln."
        )
    return train, test, len(older) - len(train)


def load_snapshot(location_id):
    project = hopsworks.login(
        host=required("HOPSWORKS_HOST"),
        project=required("HOPSWORKS_PROJECT"),
        api_key_value=required("HOPSWORKS_API_KEY"),
    )
    store = project.get_feature_store()
    try:
        group = store.get_feature_group(GROUP_NAME, version=GROUP_VERSION)
    except RestAPIError as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) == 404:
            raise NotReady(
                "weather_training_features Version 2 fehlt noch. Collector weiterlaufen lassen."
            ) from None
        raise
    if group is None:
        raise NotReady(
            "weather_training_features Version 2 fehlt noch. Collector weiterlaufen lassen."
        )
    # Preflight: don't create dataset versions while too few real rows exist.
    preview = normalize(group.read(), location_id)
    split_data(preview)
    view = store.get_or_create_feature_view(
        name=VIEW_NAME,
        version=VIEW_VERSION,
        description="3h temperature prediction from real weather; chronological local split with label gap.",
        query=group.select(COLUMNS),
        labels=[TARGET],
        training_helper_columns=["target_observed_at"],
    )
    version, _ = view.create_training_data(
        description="Real observations for one tracked weather training run.",
        data_format="parquet",
        write_options={"wait_for_job": True},
    )
    x, y = view.get_training_data(
        training_dataset_version=version,
        primary_key=True,
        event_time=True,
        training_helper_columns=True,
    )
    if y is None or TARGET not in y.columns or len(x) != len(y):
        raise ValueError("Hopsworks hat keine passenden Labels zurückgegeben.")
    # Hopsworks returns features and labels in corresponding row order.
    frame = x.reset_index(drop=True).copy()
    frame[TARGET] = y[TARGET].reset_index(drop=True)
    return normalize(frame, location_id), int(version)


def scores(y, predictions, prefix):
    return {
        f"{prefix}_mae_c": float(mean_absolute_error(y, predictions)),
        f"{prefix}_rmse_c": float(math.sqrt(mean_squared_error(y, predictions))),
    }


def iso(milliseconds):
    return pd.to_datetime(milliseconds, unit="ms", utc=True).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Positive Ridge regularization parameter.",
    )
    parser.add_argument("--location-id", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha muss positiv und endlich sein.")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").rstrip("/")
    try:
        response = requests.get(tracking_uri + "/health", timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        raise SystemExit(
            "MLflow nicht erreichbar. Zuerst: docker compose up -d mlflow"
        ) from None
    mlflow.set_tracking_uri(tracking_uri)

    try:
        frame, dataset_version = load_snapshot(args.location_id)
        train, test, excluded = split_data(frame)
    except NotReady as exc:
        print(f"Noch nicht bereit: {exc}")
        print("Kein Modell trainiert und kein MLflow-Trainingslauf erzeugt.")
        return 2

    mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", "windisch_temperature_3h")
    )
    model = make_pipeline(StandardScaler(), Ridge(alpha=args.alpha))
    with mlflow.start_run(run_name="ridge_3h") as run:
        run_id = run.info.run_id
        out = ROOT / "models" / run_id
        out.mkdir(parents=True, exist_ok=False)
        mlflow.set_tags(
            {
                "data_origin": "real_openweather",
                "location_id": frame.iloc[0]["location_id"],
                "split": "chronological_with_label_gap",
                "evaluation": "small_sample_demo",
            }
        )
        mlflow.log_params(
            {
                "algorithm": "StandardScaler + Ridge",
                "alpha": args.alpha,
                "model_features": ",".join(MODEL_FEATURES),
                "aggregation_hours": 3,
                "forecast_hours": 3,
                "label_tolerance_minutes": 30,
                "feature_group": GROUP_NAME,
                "feature_group_version": GROUP_VERSION,
                "feature_view": VIEW_NAME,
                "feature_view_version": VIEW_VERSION,
                "training_dataset_version": dataset_version,
                "total_rows": len(frame),
                "train_rows": len(train),
                "test_rows": len(test),
                "purged_rows": excluded,
                "train_start": iso(train["observed_at"].min()),
                "train_end": iso(train["observed_at"].max()),
                "test_start": iso(test["observed_at"].min()),
                "test_end": iso(test["observed_at"].max()),
            }
        )
        x_train, x_test = train[MODEL_FEATURES], test[MODEL_FEATURES]
        y_train, y_test = train[TARGET], test[TARGET]
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        baseline = x_test["temperature_current"].to_numpy()
        metrics = {
            **scores(y_test, predictions, "model"),
            **scores(y_test, baseline, "baseline"),
        }
        mlflow.log_metrics(metrics)
        frame.to_csv(out / "dataset.csv", index=False)
        train.to_csv(out / "train.csv", index=False)
        test.to_csv(out / "test.csv", index=False)
        comparison = test[["observed_at", "target_observed_at", TARGET]].copy()
        comparison["prediction_c"] = predictions
        comparison["baseline_c"] = baseline
        comparison.to_csv(out / "predictions.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        times = pd.to_datetime(test["observed_at"], unit="ms", utc=True)
        ax.plot(times, y_test, "o-", label="Beobachtet ca. 3h später")
        ax.plot(times, predictions, "o-", label="Modell")
        ax.plot(times, baseline, "--", label="Referenz: aktuelle Temperatur")
        ax.set(
            xlabel="Vorhersagezeitpunkt (UTC)",
            ylabel="Temperatur (°C)",
            title="Zeitlich getrennter Testdatensatz",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / "evaluation.png", dpi=150)
        plt.close(fig)
        versions = {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        }
        (out / "requirements.freeze.txt").write_text(
            "\n".join(f"{k}=={v}" for k, v in sorted(versions.items())) + "\n",
            encoding="utf-8",
        )
        for filename in ["training_pipeline.py", "weather_features.py"]:
            shutil.copy2(ROOT / "src" / filename, out / filename)
        dataset_hash = hashlib.sha256((out / "dataset.csv").read_bytes()).hexdigest()
        signature = mlflow.models.infer_signature(x_train, model.predict(x_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=x_train.head(3),
            serialization_format="cloudpickle",
            pip_requirements=[
                f"scikit-learn=={versions['scikit-learn']}",
                f"numpy=={versions['numpy']}",
                f"pandas=={versions['pandas']}",
                f"cloudpickle=={versions['cloudpickle']}",
            ],
        )
        metadata = {
            "run_id": run_id,
            "mlflow_model_uri": model_info.model_uri,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "location_id": str(frame.iloc[0]["location_id"]),
            "model_features": MODEL_FEATURES,
            "target": TARGET,
            "feature_group_version": GROUP_VERSION,
            "feature_view": VIEW_NAME,
            "feature_view_version": VIEW_VERSION,
            "training_dataset_version": dataset_version,
            "dataset_sha256": dataset_hash,
            "metrics": metrics,
            "train_rows": len(train),
            "test_rows": len(test),
            "purged_rows": excluded,
            "python_version": sys.version,
            "trained_through": iso(train["target_observed_at"].max()),
            "limitation": "Small time-dependent sample; no claim of robust weather forecast accuracy.",
        }
        (out / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        joblib.dump(model, out / "model.joblib")
        restored = joblib.load(out / "model.joblib")
        if not np.allclose(restored.predict(x_test), predictions):
            raise RuntimeError("Gespeichertes Modell liefert abweichende Vorhersagen.")
        mlflow.log_artifacts(str(out), artifact_path="evaluation_and_export")
    # Publish a pointer only after the whole tracked run has completed.
    pointer = ROOT / "models" / "latest.json"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_path": f"{run_id}/model.joblib",
                "metadata_path": f"{run_id}/metadata.json",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(pointer)
    print(f"Training abgeschlossen. Run: {run_id}")
    print(
        f"MAE Modell: {metrics['model_mae_c']:.3f} °C | Referenz: {metrics['baseline_mae_c']:.3f} °C"
    )
    print(f"Modell und Dokumentation: {out}")
    print("MLflow-Oberfläche: http://localhost:5001")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
