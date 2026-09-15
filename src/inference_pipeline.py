"""Predict Windisch temperature using a local model, Hopsworks and live weather."""

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import hopsworks
import joblib
import pandas as pd
from hopsworks.client.exceptions import RestAPIError

from feature_pipeline import fetch_weather, required, load_observations
from weather_features import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MAX_AGGREGATE_AGE = timedelta(minutes=90)
MAX_WEATHER_AGE = timedelta(minutes=30)
VIEW_NAME = "weather_temperature_inference"
VERSION = 2


class NotReady(Exception):
    pass


def model_file(relative_path):
    path = (MODEL_DIR / relative_path).resolve()
    if not path.is_relative_to(MODEL_DIR.resolve()):
        raise ValueError("Ungültiger Modellpfad in latest.json.")
    if not path.is_file():
        raise NotReady(f"Modelldatei fehlt: {path.name}. Zuerst Training durchführen.")
    return path


def load_model():
    pointer_path = MODEL_DIR / "latest.json"
    if not pointer_path.is_file():
        raise NotReady(
            "Noch kein Modell vorhanden. Zuerst training_pipeline.py erfolgreich ausführen."
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        model_file(pointer["metadata_path"]).read_text(encoding="utf-8")
    )
    if metadata["model_features"] != MODEL_FEATURES:
        raise ValueError(
            "Modell und aktuelle Feature-Reihenfolge passen nicht zusammen."
        )
    if metadata["feature_group_version"] != VERSION:
        raise ValueError(
            "Das Modell wurde nicht mit der Drei-Stunden-Version trainiert."
        )
    if metadata["run_id"] != pointer["run_id"]:
        raise ValueError(
            "Modellverweis und Metadaten gehören zu unterschiedlichen Runs."
        )
    model = joblib.load(model_file(pointer["model_path"]))
    if list(getattr(model, "feature_names_in_", [])) != MODEL_FEATURES:
        raise ValueError("Das gespeicherte Modell erwartet andere Eingabespalten.")
    return model, metadata


def get_aggregate(location_id, weather_time):
    project = hopsworks.login(
        host=required("HOPSWORKS_HOST"),
        project=required("HOPSWORKS_PROJECT"),
        api_key_value=required("HOPSWORKS_API_KEY"),
    )
    store = project.get_feature_store()
    try:
        group = store.get_feature_group("weather_features", version=VERSION)
    except RestAPIError as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) == 404:
            raise NotReady(
                "weather_features Version 2 fehlt noch. Collector weiterlaufen lassen."
            ) from None
        raise
    if group is None:
        raise NotReady(
            "weather_features Version 2 fehlt noch. Collector weiterlaufen lassen."
        )
    view = store.get_or_create_feature_view(
        name=VIEW_NAME,
        version=VERSION,
        description="Latest available 3h temperature aggregate; current RT temperature is fetched at inference.",
        query=group.select(["location_id", "observed_at", "temperature_mean_3h"]),
    )
    frame = view.get_batch_data(
        primary_key=True,
        event_time=True,
        dataframe_type="pandas",
        transformed=False,
    )
    print(f"Aus Feature View geladene Zeilen: {len(frame)}")
    if frame.empty:
        raise NotReady(
            "Keine aktuellen aggregierten Features vorhanden. Feature-Pipeline ausführen."
        )
    frame = frame.loc[frame["location_id"].astype(str) == location_id].copy()
    if frame.empty:
        raise NotReady("Keine aktuellen Features für den gewählten Standort vorhanden.")
    if pd.api.types.is_numeric_dtype(frame["observed_at"]):
        frame["feature_time"] = pd.to_datetime(
            frame["observed_at"], unit="ms", utc=True
        )
    else:
        frame["feature_time"] = pd.to_datetime(frame["observed_at"], utc=True)
    # Never choose a feature timestamp later than the live observation.
    frame = frame.loc[frame["feature_time"] <= weather_time]
    if frame.empty:
        raise NotReady("Kein Feature mit passendem Zeitpunkt verfügbar.")
    row = frame.sort_values("feature_time").iloc[-1]
    feature_time = row["feature_time"].to_pydatetime()
    print(f"Wetterzeitpunkt: {weather_time.isoformat()}")
    print(f"Aggregat-Zeitpunkt: {feature_time.isoformat()}")
    print(f"Abstand: {(weather_time - feature_time).total_seconds() / 60:.1f} Minuten")
    print(f"Originalwert observed_at: {row['observed_at']}")
    if weather_time - feature_time > MAX_AGGREGATE_AGE:
        raise NotReady(
            "Das aggregierte Feature ist zu alt. Feature-Pipeline aktualisieren."
        )
    value = float(row["temperature_mean_3h"])
    if not math.isfinite(value):
        raise ValueError("Ungültiger aggregierter Temperaturwert.")
    return value, feature_time


def replay_weather(data_dir, at, metadata):
    """Replay one genuine held-out observation; never generate weather values."""
    frame = load_observations(data_dir)
    frame = frame.loc[frame["location_id"] == metadata["location_id"]].copy()
    trained_through = pd.Timestamp(metadata["trained_through"])
    if trained_through.tzinfo is None:
        raise ValueError("Trainingsende in metadata.json muss eine Zeitzone enthalten.")
    # Replay must be later than every label used to fit this model.
    trained_ms = trained_through.value // 1_000_000
    frame = frame.loc[frame["observed_at"] > trained_ms]
    if at:
        timestamp = pd.Timestamp(at)
        if timestamp.tzinfo is None:
            raise ValueError(
                "--at benötigt einen ISO-Zeitstempel mit Zeitzone, z.B. ...+00:00."
            )
        frame = frame.loc[frame["observed_at"] == timestamp.value // 1_000_000]
    if frame.empty:
        raise NotReady(
            "Keine passende echte Replay-Beobachtung nach dem Trainingsende vorhanden."
        )
    row = frame.sort_values("observed_at").iloc[-1].to_dict()
    row["observed_at"] = pd.to_datetime(
        row["observed_at"], unit="ms", utc=True
    ).isoformat()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Echte historische Beobachtung als damaligen RT-Wert verwenden; kein OpenWeather-Aufruf.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "sample",
        help="JSON-Quellordner für Replay.",
    )
    parser.add_argument(
        "--at",
        default=None,
        help="Exakter gespeicherter Wetterzeitpunkt mit Zeitzone; nur mit --replay.",
    )
    args = parser.parse_args()
    if args.at and not args.replay:
        parser.error("--at ist nur mit --replay erlaubt.")
    model, metadata = load_model()
    mode = "replay" if args.replay else "live"
    weather = (
        replay_weather(args.data_dir.resolve(), args.at, metadata)
        if args.replay
        else fetch_weather()
    )
    weather_time = datetime.fromisoformat(weather["observed_at"])
    location_id = f"{weather['latitude']:.6f}_{weather['longitude']:.6f}"
    if location_id != metadata["location_id"]:
        raise ValueError("Modell und abgefragter Wetterstandort stimmen nicht überein.")
    aggregate, feature_time = get_aggregate(location_id, weather_time)
    now = datetime.now(timezone.utc)
    weather_age = now - weather_time
    if not args.replay and (
        weather_age < timedelta(minutes=-5) or weather_age > MAX_WEATHER_AGE
    ):
        raise NotReady(
            "OpenWeather-Wetterzeitpunkt ist nicht aktuell genug (maximal 30 Minuten)."
        )
    current_temperature = float(weather["temperature_c"])
    if not math.isfinite(current_temperature):
        raise ValueError("Ungültige aktuelle Temperatur.")
    inputs = pd.DataFrame(
        [
            {
                "temperature_mean_3h": aggregate,
                "temperature_current": current_temperature,
            }
        ],
        columns=MODEL_FEATURES,
    ).astype("float64")
    prediction = float(model.predict(inputs)[0])
    if not math.isfinite(prediction):
        raise ValueError("Modell hat keine gültige Temperatur geliefert.")
    target_time = weather_time + timedelta(hours=3)
    result = {
        "mode": mode,
        "generated_at": now.isoformat(),
        "city": weather["city"],
        "country": weather["country"],
        "location_id": location_id,
        "model_run_id": metadata["run_id"],
        "weather_observed_at": weather_time.isoformat(),
        "aggregate_observed_at": feature_time.isoformat(),
        "aggregate_lag_minutes": (weather_time - feature_time).total_seconds() / 60,
        "feature_view": VIEW_NAME,
        "feature_view_version": VERSION,
        "model_inputs": inputs.iloc[0].to_dict(),
        "target_time": target_time.isoformat(),
        "predicted_temperature_c": prediction,
    }
    output_dir = ROOT / "data" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{mode}_prediction_{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    local_zone = ZoneInfo("Europe/Zurich")
    if args.replay:
        print("REPLAY: echte historische Beobachtung; keine aktuelle Wetterprognose.")
    else:
        print("LIVE: frisch abgerufene Wetterdaten.")
    print(f"Ort: {weather['city']}, {weather['country']}")
    print(f"Temperatur am Vorhersagezeitpunkt: {current_temperature:.2f} °C")
    print(f"Gespeicherter 3h-Durchschnitt: {aggregate:.2f} °C")
    print(
        f"Abstand Aggregat zum Wetterzeitpunkt: {result['aggregate_lag_minutes']:.1f} Minuten"
    )
    print(
        f"Stand des Aggregats: {feature_time.astimezone(local_zone):%d.%m.%Y %H:%M %Z}"
    )
    print(
        f"Vorhersage für {target_time.astimezone(local_zone):%d.%m.%Y %H:%M %Z}: {prediction:.2f} °C"
    )
    print(f"Modell-Run: {metadata['run_id']}")
    print(f"Gespeichert: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotReady as exc:
        print(f"Noch nicht bereit: {exc}")
        raise SystemExit(2)
