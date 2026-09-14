"""Collect current weather and upsert locally collected observations to Hopsworks."""

import argparse
import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv
from weather_features import prepare_features

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
load_dotenv(PROJECT_DIR / ".env", override=True)


def required(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Fehlender Eintrag in .env: {name}")
    return value


def api_get(url, params):
    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.RequestException:
        raise SystemExit("OpenWeather nicht erreichbar. Netzwerk prüfen.") from None
    if response.status_code != 200:
        raise SystemExit(
            f"OpenWeather HTTP {response.status_code}. Bei 401 API-Key prüfen."
        )
    try:
        return response.json()
    except ValueError:
        raise SystemExit("OpenWeather hat kein gültiges JSON geliefert.") from None


def fetch_weather():
    api_key = required("OPENWEATHER_API_KEY")
    city = os.getenv("WEATHER_CITY", "Windisch").strip()
    country = os.getenv("WEATHER_COUNTRY", "CH").strip().upper()
    locations = api_get(
        "https://api.openweathermap.org/geo/1.0/direct",
        {"q": f"{city},{country}", "limit": 5, "appid": api_key},
    )
    matches = [item for item in locations if item.get("country") == country]
    if len(matches) != 1:
        raise SystemExit(
            f"Ort nicht eindeutig gefunden: {city}, {country} ({len(matches)} Treffer)."
        )
    location = matches[0]
    weather = api_get(
        "https://api.openweathermap.org/data/2.5/weather",
        {
            "lat": location["lat"],
            "lon": location["lon"],
            "appid": api_key,
            "units": "metric",
        },
    )
    return {
        "city": location["name"],
        "country": location["country"],
        "latitude": location["lat"],
        "longitude": location["lon"],
        "observed_at": datetime.fromtimestamp(
            weather["dt"], tz=timezone.utc
        ).isoformat(),
        "temperature_c": weather["main"]["temp"],
        "humidity_pct": weather["main"]["humidity"],
        "pressure_hpa": weather["main"]["pressure"],
        "wind_speed_ms": weather["wind"]["speed"],
        "cloud_cover_pct": weather["clouds"]["all"],
    }


def save_observation(observation):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.fromisoformat(observation["observed_at"]).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    path = (
        RAW_DIR
        / f"{observation['latitude']}_{observation['longitude']}_{timestamp}.json"
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(observation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    print(f"Ort: {observation['city']}, {observation['country']}", flush=True)
    print(f"Temperatur: {observation['temperature_c']} °C", flush=True)
    print(f"Lokal gespeichert: {path}", flush=True)


def load_observations(data_dir):
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"Keine Beobachtungen in {data_dir} vorhanden.")
    rows = []
    for path in files:
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            raise SystemExit(f"Datei kann nicht gelesen werden: {path.name}") from None
    frame = pd.DataFrame(rows)
    numeric = [
        "latitude",
        "longitude",
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_speed_ms",
        "cloud_cover_pct",
    ]
    columns = ["city", "country", "observed_at", *numeric]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise SystemExit(f"Fehlende Datenfelder: {', '.join(sorted(missing))}")
    frame = frame[columns].copy()
    if frame.isna().any().any():
        raise SystemExit("Unvollständige Beobachtungen gefunden. JSON-Dateien prüfen.")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    if not all(math.isfinite(value) for value in frame[numeric].to_numpy().flat):
        raise SystemExit("Nicht-endliche Messwerte gefunden. Quelldaten prüfen.")
    for column in ["city", "country"]:
        frame[column] = frame[column].astype(str)
    # Hopsworks interprets integer event times as Unix milliseconds.
    timestamps = pd.to_datetime(frame["observed_at"], utc=True, errors="raise")
    frame["observed_at"] = timestamps.map(
        lambda value: value.value // 1_000_000
    ).astype("int64")
    frame["location_id"] = frame.apply(
        lambda row: f"{row['latitude']:.6f}_{row['longitude']:.6f}", axis=1
    )
    frame = frame.drop_duplicates(["location_id", "observed_at"], keep="last")
    return frame.sort_values(["location_id", "observed_at"]).reset_index(drop=True)


def upload_all(observations, features, training):
    project = hopsworks.login(
        host=required("HOPSWORKS_HOST"),
        project=required("HOPSWORKS_PROJECT"),
        api_key_value=required("HOPSWORKS_API_KEY"),
    )
    feature_store = project.get_feature_store()
    groups = [
        (
            "weather_observations",
            1,
            observations,
            "OpenWeather observations. observed_at is Unix time in milliseconds, UTC.",
        ),
        (
            "weather_features",
            2,
            features,
            "Real weather features: prior 3h mean of hourly means (3/3 slots, gaps <=2h), "
            "plus temperature at prediction time. All timestamps are UTC Unix milliseconds.",
        ),
        (
            "weather_training_features",
            2,
            training,
            "Weather features with observed temperature at +3h (+/-30min). "
            "Only mature labels; no generated or interpolated weather values.",
        ),
    ]
    for name, version, frame, description in groups:
        if frame.empty:
            print(
                f"{name}: Noch keine ausreichenden echten Daten; Upload übersprungen.",
                flush=True,
            )
            continue
        group = feature_store.get_or_create_feature_group(
            name=name,
            version=version,
            description=description,
            primary_key=["location_id", "observed_at"],
            event_time="observed_at",
            online_enabled=True,
        )
        print(f"Übertrage {len(frame)} Zeilen nach {name} ...", flush=True)
        group.insert(frame, operation="upsert", write_options={"wait_for_job": True})
        print(f"Hopsworks-Insert abgeschlossen: {name}, Version {version}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--upload-only",
        action="store_true",
        help="Lokale Daten aufbereiten und hochladen; kein OpenWeather-Abruf.",
    )
    modes.add_argument(
        "--prepare-only",
        action="store_true",
        help="Nur lokale Daten aufbereiten und anzeigen; keine API-Aufrufe.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DIR,
        help="JSON-Quellordner, z.B. data/sample. Nur mit einem der obigen Modi.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if not (args.upload_only or args.prepare_only):
        if data_dir != RAW_DIR.resolve():
            parser.error(
                "--data-dir ist nur mit --upload-only oder --prepare-only erlaubt."
            )
        save_observation(fetch_weather())
    observations = load_observations(data_dir)
    features, training = prepare_features(observations)
    print(f"Beobachtungen: {len(observations)}", flush=True)
    for location_id, group in observations.groupby("location_id"):
        span = (group["observed_at"].max() - group["observed_at"].min()) / 3_600_000
        print(
            f"Standort {location_id}: gesammelte Zeitspanne {span:.1f} Stunden.",
            flush=True,
        )
    print(f"Gültige Feature-Zeilen: {len(features)}", flush=True)
    print(f"Trainingszeilen mit echtem Label: {len(training)}", flush=True)
    if features.empty:
        print(
            "Warte auf 3h Historie mit allen 3 belegten Stundenfenstern "
            "und ohne Lücken über 2h.",
            flush=True,
        )
    elif training.empty:
        print(
            "Features vorhanden. Warte auf passende spätere Temperaturen "
            "und Ablauf des Label-Zeitfensters (+3h30min).",
            flush=True,
        )
    if args.prepare_only:
        if not features.empty:
            print("Letzte Feature-Zeilen (Zeitstempel in UTC-Millisekunden):")
            print(features.tail(5).to_string(index=False))
        if not training.empty:
            print("Letzte Trainingszeilen:")
            print(training.tail(5).to_string(index=False))
        return
    upload_all(observations, features, training)


if __name__ == "__main__":
    main()
