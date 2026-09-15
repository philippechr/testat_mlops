# MLOps-Testat: Temperaturvorhersage für Windisch

Dieses Projekt sagt die Temperatur in Windisch (CH) ungefähr drei Stunden im Voraus vorher. Es implementiert eine Feature–Training–Inference-Architektur mit OpenWeather als Datenquelle, Hopsworks als Featurestore und MLflow für Experiment-Tracking.

## Einstieg für die Bewertung

**Der Bewertungsablauf verwendet die echten historischen Beispieldaten unter `data/sample/`. Der Collector muss dafür nicht gestartet werden. Es sind weder eine Sammelphase noch ein OpenWeather API-Key erforderlich.**

Die folgenden Schritte führen lokal durch Datenimport, Training und historisches Replay. Docker führt die Python-Skripte und MLflow auf dem eigenen Rechner aus; Hopsworks bleibt ein externer Cloud-Dienst. Internetzugang ist daher erforderlich.

### 1. Voraussetzungen und Einrichtung

Benötigt werden Git, Docker mit laufender Linux-Engine, Docker Compose ab Version 2.24.0 sowie ein eigenes Hopsworks-Projekt mit API-Key. Für eine isolierte Reproduktion ein frisches Projekt verwenden.

Die Terminalbefehle sind für macOS/Linux beziehungsweise Git Bash vorgesehen:

```bash
git clone https://github.com/philippechr/testat_mlops.git
cd testat_mlops
cp .env.example .env
```

Falls noch nicht erfolgt, die Datei `.env.example` in `.env` umbenennen und Zugangsdaten (HOPSWORKS_API_KEY,  OPENWEATHER_API_KEY) aus MS-Teams-Abgabe eintragen. Für Replay bleibt der OpenWeather-Key leer:

```dotenv
OPENWEATHER_API_KEY=siehe_msTeams <--------------
WEATHER_CITY=Windisch
WEATHER_COUNTRY=CH
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PROJECT=testat_fhnw_mlops
HOPSWORKS_API_KEY=siehe_msTeams <--------------
```

Python muss lokal nicht installiert sein: Python 3.12 und die direkten Paketversionen sind in `compose.yaml` festgelegt. Das Image verwendet `linux/amd64`, auf Apple Silicon mit Emulation. Der Hopsworks-Client muss zur verwendeten Plattformversion passen.

### 2. Image bauen und MLflow starten

Alle folgenden Befehle im Repository-Ordner ausführen:

```bash
docker compose config --quiet
docker compose build
docker compose up -d mlflow
docker compose ps
```

Warten, bis MLflow `healthy` ist. Falls nötig, `docker compose ps` erneut ausführen. Die Weboberfläche ist unter **http://localhost:5001** erreichbar. Der Collector bleibt bei diesem Start ausgeschaltet.

### 3. Verbindung prüfen und Beispieldaten importieren

```bash
docker compose run --rm python python src/check_hopsworks.py

docker compose run --rm python python src/feature_pipeline.py --upload-only --data-dir /app/data/sample
```

Die Feature-Pipeline liest die echten Beispieldaten, berechnet Aggregationen und Labels und speichert die Dataframes in Hopsworks. `--upload-only` verhindert einen neuen OpenWeather-Abruf.

### 4. Modell trainieren und evaluieren

```bash
docker compose run --rm python python src/training_pipeline.py
```

Erwartet werden ein erfolgreicher Run im MLflow-Experiment **`windisch_temperature_3h`** und ein Modell unter `models/<run_id>/`. `models/latest.json` verweist auf diesen Run.

### 5. Historische Inferenz ausführen

```bash
docker compose run --rm python python src/inference_pipeline.py --replay --data-dir /app/data/sample --at "2026-09-15T16:25:05+00:00"
```

Die Vorhersage wird direkt im Terminal angezeigt und zusätzlich als JSON unter `data/predictions/` gespeichert.
Die zuletzt gespeicherte Vorhersage lässt sich so erneut anzeigen:
```bash
docker compose run --rm python python -c "from pathlib import Path; import json; files = sorted(Path('/app/data/predictions').glob('replay_prediction_*.json')); print(json.dumps(json.loads(files[-1].read_text()), indent=2, ensure_ascii=False) if files else 'Noch keine Replay-Vorhersage vorhanden.')"
```


Replay verwendet eine echte gespeicherte Beobachtung nach dem Trainingszeitraum als damaligen aktuellen Temperaturwert und lädt das passende Aggregat über eine Hopsworks Feature View. Es ruft OpenWeather nicht auf und erzeugt keine künstlichen Temperaturen.

Die Ausgabe ist als **Replay** gekennzeichnet, enthält eine historische Vorhersage und wird unter `data/predictions/` gespeichert. Sie ist keine heutige Live-Vorhersage.

Nach dem Datenimport kann alternativ das mitgelieferte Modell direkt für Replay verwendet werden. Für den vollständigen Pipeline-Nachweis gehört auch Schritt 4 dazu.

### 6. Beenden

```bash
docker compose down
```

Ohne `-v` bleibt das MLflow-Volume erhalten. Beispieldaten und Modelle liegen im lokalen Projektordner und bleiben ebenfalls erhalten.

## Architektur und Dateien

```text
OpenWeather / Beispieldaten → Feature-Pipeline → Hopsworks
                                                   ↓
                                            Training-Pipeline
                                                   ↓
                                             Modell + MLflow
                                                   ↓
Hopsworks-Aggregat + aktueller Temperaturwert → Inferenz
```

| Datei | Aufgabe |
|---|---|
| `src/collector.py` | Regelmässiger Aufruf der Feature-Pipeline für die Live-Sammlung |
| `src/feature_pipeline.py` | Daten einlesen/abrufen, Features und Labels erstellen, nach Hopsworks schreiben |
| `src/weather_features.py` | Gemeinsame Berechnungsregeln |
| `src/training_pipeline.py` | Trainingsdatensatz über eine Feature View erstellen, Modell trainieren und evaluieren |
| `src/inference_pipeline.py` | Live-Inferenz oder historisches Replay |
| `data/sample/` | Fester Beispieldatensatz aus echten OpenWeather-Beobachtungen |
| `models/` | Modell, Metadaten und Trainingsdokumentation |

## Daten, Features und Modell

OpenWeather liefert Temperatur, Luftfeuchtigkeit, Luftdruck, Windgeschwindigkeit und Bewölkung. Historische Daten entstanden durch die Speicherung aktueller Beobachtungen. Verarbeitet werden die Wetterzeitpunkte der API in UTC.

Das Modell verwendet genau zwei Eingaben:

- **`temperature_mean_3h` – aggregiert:** Mittelwert der stündlichen Temperaturmittelwerte im Intervall `[t−3h, t)`. Alle drei Stundenfenster müssen echte Beobachtungen enthalten. Die Historie muss bis mindestens `t−3h` zurückreichen; Lücken über zwei Stunden sind nicht zulässig. Fehlende Werte werden nicht aufgefüllt.
- **`temperature_current` – RT-Feature:** Temperatur zum Vorhersagezeitpunkt. Bei Live-Inferenz frisch von OpenWeather abgerufen; beim Training und Replay die entsprechende historische Beobachtung.

Das Label **`temperature_target_3h`** ist die echte Temperaturbeobachtung, die dem Zeitpunkt `t+3h` am nächsten liegt, mit einer Toleranz von ±30 Minuten. Es wird erst erstellt, wenn eine Beobachtung bei oder nach `t+3h30min` vorliegt.

Das Modell besteht aus `StandardScaler → Ridge-Regression` mit `alpha=1.0`. Die Daten werden chronologisch ungefähr 80/20 aufgeteilt. Trainingszeilen, deren Label-Zeitfenster bis in den Testzeitraum reicht, werden ausgeschlossen. Der Scaler wird nur auf den Trainingsdaten angepasst.

Die Evaluation verwendet **MAE und RMSE in °C**. Als Referenz dient „Temperatur in drei Stunden = aktuelle Temperatur“. Modell und Referenz werden auf denselben Testzeilen bewertet; das exportierte Modell wird nicht nachträglich auf dem Testdatensatz trainiert.

## Hopsworks, Tracking und Modellbereitstellung

| Feature Group | Version | Inhalt |
|---|---:|---|
| `weather_observations` | 1 | Wetterbeobachtungen |
| `weather_features` | 2 | Gültige Drei-Stunden-Features |
| `weather_training_features` | 2 | Features mit vorhandenem Label |

Primärschlüssel sind `location_id` und `observed_at`; Event Time ist `observed_at`. Wiederholte Uploads verwenden Upserts. Die Trainings-Feature-View wählt Features und Label aus und erzeugt einen versionierten Parquet-Trainingsdatensatz.

MLflow protokolliert Parameter, Metriken, Zeiträume, Datenversionen und Modellartefakte. Jeder erfolgreiche Run exportiert unter `models/<run_id>/`:

- `model.joblib` und `metadata.json`
- Datensnapshot, Trainings-/Testdaten und Vorhersagen
- `evaluation.png`
- `requirements.freeze.txt` und verwendete Skripte

Die Inferenz lädt das Modell lokal über `models/latest.json`. Dies ist laut Aufgabenstellung zulässig. MLflow dient dem Tracking und der Speicherung von Modellartefakten; eine zusätzliche Model Registry ist nicht implementiert.

MLflow speichert seine Daten in einem Docker-Volume. Dieses wird nicht über Git übertragen. Das mitgelieferte Modell und erneutes Training sind davon unabhängig.

## Optional: Live-Sammlung und Live-Inferenz
„Optional“ bezieht sich auf die Wiederholung durch den Dozenten. Der Bewertungsablauf mit Beispieldaten benötigt keine laufende Datensammlung.

Die Verwendung eines frisch abgerufenen RT-Features gehört zur Projektumsetzung und wird durch einen erfolgreichen Live-Durchlauf nachgewiesen. Replay ersetzt diesen Nachweis nicht.

Einen gültigen `OPENWEATHER_API_KEY` in `.env` eintragen und den Collector starten:

```bash
docker compose up -d python
docker compose logs -f --tail=30 python
```

Der Collector ruft sofort Daten ab und wartet nach jedem erfolgreichen Durchlauf 30 Minuten. Bei Fehlern versucht er es nach fünf Minuten erneut. Ein Durchlauf umfasst zwei OpenWeather-Anfragen: Ortssuche und Wetterabruf. Die Pipeline-Laufzeit kommt zur Wartezeit hinzu; Neustarts lösen sofort einen Abruf aus.

Docker und der Rechner müssen laufen, wach bleiben und Internetzugang haben. Rohdaten liegen unter `data/raw/`; gleiche Wetterzeitpunkte werden nicht doppelt gezählt. `Ctrl+C` beendet nur die Log-Anzeige.

Sobald ein trainiertes Modell und ausreichend aktuelle Features in Hopsworks vorhanden sind:

```bash
docker compose exec python python src/inference_pipeline.py
```

Die Inferenz verwendet das neueste passende Aggregat aus Hopsworks und die frisch abgerufene Temperatur gemeinsam. Die Wetterbeobachtung darf höchstens 30 Minuten alt sein; das Aggregat darf maximal 90 Minuten hinter ihr liegen. Beide Bezugszeitpunkte werden dokumentiert. Der Vorhersagehorizont beginnt beim Wetterbeobachtungszeitpunkt.

Alte Beispieldaten allein reichen wegen dieser Aktualitätsprüfungen nicht für Live-Inferenz. Fehlen aktuelle Features im eigenen Hopsworks-Projekt, muss zunächst gesammelt werden.

## Einschränkungen und aufgetretene Schwierigkeiten

- Ein Standort und ein kurzer Sammelzeitraum ermöglichen keine belastbare Aussage über saisonale Vorhersagequalität.
- Messlücken können Feature- und Trainingszeilen verhindern. Der Collector wurde auf 30 Minuten umgestellt, um die drei Stundenfenster zuverlässiger abzudecken.
- Labels sind um ±30 Minuten zeitlich toleriert. Das Aggregat ist ein Mittel aus Stichproben, kein kontinuierlicher Temperaturmittelwert.
- Bei Live-Inferenz kann das Aggregat älter sein als die aktuelle Temperatur; beim Training sind beide auf denselben historischen Zeitpunkt bezogen. Diese Abweichung wird durch die Aktualitätsgrenze begrenzt und in der Ausgabe dokumentiert.
- Das Training verlangt mindestens 24 vollständige Zeilen sowie zwölf Trainings- und fünf Testzeilen nach dem zeitlichen Ausschluss. Das sind technische Mindestwerte, keine statistische Qualitätsgarantie.
- Alle lokalen Daten werden bei jedem Feature-Durchlauf erneut verarbeitet und hochgeladen; das ist für kleine Datensätze einfach, aber nicht für grosse Datenmengen optimiert.
- Die anfänglichen ARM-Build-Probleme wurden durch `linux/amd64`, das passende Binärpaket und benötigte Systempakete behoben.
- Direkte Paketversionen sind festgelegt; Basisimage und transitive Abhängigkeiten sind nicht vollständig eingefroren.
- Im getesteten Lauf betrug der MAE des Modells 6,731 °C gegenüber 3,061 °C der Persistenzreferenz. Das Modell war damit schlechter als die einfache Referenz.
- Für die neueste Sample-Beobachtung war kein ausreichend zeitnahes Aggregat verfügbar. Der dokumentierte Replay-Befehl verwendet deshalb einen expliziten historischen Zeitpunkt mit passendem Aggregat.
- Die Inferenz lädt die gesamte Feature View und filtert Standort und Zeitpunkt lokal. Das ist für den kleinen Beispieldatensatz praktikabel, skaliert aber nicht für grosse Datenmengen.

## Referenzen

- [OpenWeather API](https://openweathermap.org/api)
- [Hopsworks Feature Groups](https://docs.hopsworks.ai/latest/python-api/hsfs/feature_group/)
- [Hopsworks Feature Views](https://docs.hopsworks.ai/latest/python-api/hsfs/feature_view/)
- [MLflow](https://mlflow.org/docs/latest/)
- Aufgabenstellung „Projektarbeit MLOps“, FHNW
