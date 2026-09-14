import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env", override=True)


def api_get(url, params):
    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.RequestException:
        raise SystemExit("OpenWeather nicht erreichbar. Netzwerk prüfen.") from None

    if response.status_code != 200:
        raise SystemExit(
            f"OpenWeather meldet HTTP {response.status_code}. "
            "Bei 401 API-Key und Freischaltung prüfen."
        )

    return response.json()


def fetch_weather():
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise SystemExit("Bitte OPENWEATHER_API_KEY in der .env setzen.")

    city = os.getenv("WEATHER_CITY", "Windisch")
    country = os.getenv("WEATHER_COUNTRY", "CH")

    # Stadtnamen in Koordinaten übersetzen.
    locations = api_get(
        "https://api.openweathermap.org/geo/1.0/direct",
        {
            "q": f"{city},{country}",
            "limit": 5,
            "appid": api_key,
        },
    )

    matches = [location for location in locations if location.get("country") == country]

    if not matches:
        raise SystemExit(f"Ort nicht gefunden: {city}, {country}")

    if len(matches) > 1:
        raise SystemExit(
            f"Mehrere Orte für {city}, {country} gefunden. "
            "Die Ortsauswahl muss genauer eingegrenzt werden."
        )

    location = matches[0]
    latitude = location["lat"]
    longitude = location["lon"]

    weather = api_get(
        "https://api.openweathermap.org/data/2.5/weather",
        {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        },
    )

    return {
        "city": location["name"],
        "country": location["country"],
        "latitude": latitude,
        "longitude": longitude,
        "observed_at": datetime.fromtimestamp(
            weather["dt"], tz=timezone.utc
        ).isoformat(),
        "temperature_c": weather["main"]["temp"],
        "humidity_pct": weather["main"]["humidity"],
        "pressure_hpa": weather["main"]["pressure"],
        "wind_speed_ms": weather["wind"]["speed"],
        "cloud_cover_pct": weather["clouds"]["all"],
    }


def main():
    observation = fetch_weather()

    output_dir = PROJECT_DIR / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.fromisoformat(observation["observed_at"]).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    filename = f"{observation['latitude']}_{observation['longitude']}_{timestamp}.json"
    output_file = output_dir / filename

    output_file.write_text(
        json.dumps(observation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Ort: {observation['city']}, {observation['country']}")
    print(f"Koordinaten: {observation['latitude']}, {observation['longitude']}")
    print(f"Zeitpunkt: {observation['observed_at']}")
    print(f"Temperatur: {observation['temperature_c']} °C")
    print(f"Gespeichert: {output_file}")


if __name__ == "__main__":
    main()
