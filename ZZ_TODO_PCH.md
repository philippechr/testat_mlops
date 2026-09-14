# To-do: MLOps-Testat abschliessen und abgeben

## 1. Genügend echte Daten sammeln

- [ ] Im bisherigen Repository den Datenstand prüfen:

```bash
docker compose exec python python src/feature_pipeline.py --prepare-only
```

Mindestens **24 vollständige Trainingszeilen** werden benötigt. Nach der zeitlichen Trennung müssen mindestens zwölf Trainings- und fünf Testzeilen verbleiben.

## 2. Beispieldatensatz fixieren

- [ ] Collector stoppen und echte Beobachtungen kopieren:

```bash
docker compose stop python
mkdir -p data/sample
cp data/raw/*.json data/sample/
```

- [ ] Beispieldaten prüfen:

```bash
docker compose run --rm python python src/feature_pipeline.py --prepare-only --data-dir /app/data/sample
```

Die Kopie ergänzt beziehungsweise überschreibt Dateien, entfernt aber keine älteren Dateien aus `data/sample/`. Bei einer späteren Neuauswahl den Inhalt bewusst prüfen.

## 3. Vollständigen Ablauf mit bestehendem Hopsworks-Projekt prüfen


```bash
docker compose up -d mlflow
docker compose ps
```

Warten, bis MLflow `healthy` ist. Dann nacheinander ausführen:

- [ ] Verbindung prüfen:

```bash
docker compose run --rm python python src/check_hopsworks.py
```

- [ ] Beispieldaten importieren:

```bash
docker compose run --rm python python src/feature_pipeline.py --upload-only --data-dir /app/data/sample
```

- [ ] Modell trainieren:

```bash
docker compose run --rm python python src/training_pipeline.py
```

- [ ] Historisches Replay ausführen:

```bash
docker compose run --rm python python src/inference_pipeline.py --replay --data-dir /app/data/sample
```

**Alle vier Befehle müssen erfolgreich sein.** Bei unzureichenden Daten weiter sammeln und den Beispieldatensatz ergänzen.

## 4. Ergebnisse kontrollieren

MLflow unter [http://localhost:5001](http://localhost:5001) öffnen.

- [ ] Erfolgreicher Trainingsrun im Experiment `windisch_temperature_3h` vorhanden.
- [ ] MAE und RMSE für Modell und Referenz vorhanden.
- [ ] Modell und Evaluationsartefakte vorhanden.
- [ ] Lokale Dateien prüfen:

```bash
cat models/latest.json
ls -la models/
ls -la data/predictions/
```

- [ ] Das in `latest.json` bezeichnete Run-Verzeichnis ist vollständig vorhanden, insbesondere mit `model.joblib` und `metadata.json`.

## 5. Echte Live-Inferenz dokumentieren

- [ ] Gültigen OpenWeather-Key in `.env` prüfen.
- [ ] Collector starten:

```bash
docker compose up -d python
docker compose logs -f --tail=30 python
```

Mit `Ctrl+C` die Log-Anzeige verlassen. Der Collector läuft weiter.

- [ ] Sobald aktuelle aggregierte Features in Hopsworks vorhanden sind, Live-Inferenz ausführen:

```bash
docker compose exec python python src/inference_pipeline.py
```

- [ ] Erfolgreiches Ergebnis dokumentieren: Zeitpunkt, Modell-Run-ID, aggregiertes Feature, frisch abgerufene Temperatur und Vorhersage.
- [ ] Diese Angaben kurz im README als Live-Nachweis ergänzen. Die JSON-Ausgabe liegt unter `data/predictions/`.

## 6. Abgabedateien auf GitHub hochladen

Folgende Dateien müssen vorhanden sein:

- [ ] `src/` mit allen Skripten
- [ ] `compose.yaml`
- [ ] `.env.example` ohne echte Schlüssel
- [ ] `.gitignore`
- [ ] Finale `README.md`
- [ ] `data/sample/`
- [ ] `models/latest.json`
- [ ] Vollständiges dazugehöriges Modellverzeichnis

Codespaces und `.devcontainer/` werden für die lokale Abgabe nicht benötigt.

- [ ] Prüfen, dass `.env` ignoriert und nicht von Git verfolgt wird:

```bash
git check-ignore .env
git ls-files .env
```

Beim ersten Befehl sollte `.env` erscheinen; beim zweiten **keine Ausgabe**. Falls der zweite Befehl `.env` anzeigt, vor dem Upload klären und aus der Git-Verfolgung entfernen. Bereits veröffentlichte Schlüssel müssen ersetzt werden.

- [ ] Dateien gezielt vorbereiten:

```bash
git add compose.yaml README.md .gitignore .env.example src/ data/sample/ models/
git diff --cached --stat
git status
```

- [ ] Vorbereiteten Inhalt auf Zugangsdaten prüfen.
- [ ] Committen und hochladen:

```bash
git commit -m "MLOps-Testat für lokale Reproduktion vorbereiten"
git push origin main
```

## 7. README aus einem frischen lokalen Klon prüfen

- [ ] Im bisherigen Repository die Container herunterfahren:

```bash
docker compose down
```

**Kein `-v` verwenden.**

- [ ] In einen neuen Ordner klonen:

```bash
cd ..
git clone https://github.com/philippechr/testat_mlops.git testat_mlops_abgabecheck
cd testat_mlops_abgabecheck
```

- [ ] Eine separate Docker-Umgebung festlegen:

```bash
export COMPOSE_PROJECT_NAME=mlops_abgabecheck
```

Alle folgenden Befehle im selben Terminal ausführen. Die separate Projektkennung verwendet eigene Container und ein eigenes MLflow-Volume. Beim erstmaligen Test ist dieses Volume leer. Für eine spätere Wiederholung mit erneut leerem Speicher eine neue Projektkennung wählen.

- [ ] Konfiguration erstellen:

```bash
cp .env.example .env
```

- [ ] **Ein weiteres frisches Hopsworks-Projekt** erstellen und dessen Zugangsdaten in diese neue `.env` eintragen. Für Beispieldaten und Replay darf der OpenWeather-Key leer bleiben.
- [ ] Frisch bauen und starten:

```bash
docker compose config --quiet
docker compose build --no-cache
docker compose up -d mlflow
docker compose ps
```

- [ ] Warten, bis MLflow `healthy` ist.
- [ ] Verbindung prüfen und Beispieldaten importieren:

```bash
docker compose run --rm python python src/check_hopsworks.py
docker compose run --rm python python src/feature_pipeline.py --upload-only --data-dir /app/data/sample
```

- [ ] **Zuerst das mitgelieferte Modell prüfen**, bevor neues Training den Modellverweis aktualisiert:

```bash
docker compose run --rm python python src/inference_pipeline.py --replay --data-dir /app/data/sample
```

- [ ] Danach Neutraining und erneutes Replay ausführen:

```bash
docker compose run --rm python python src/training_pipeline.py
docker compose run --rm python python src/inference_pipeline.py --replay --data-dir /app/data/sample
```

Damit werden sowohl das Abgabemodell als auch der vollständige Neuaufbau geprüft. Der Collector bleibt ausgeschaltet.

- [ ] Nach erfolgreichem Test beenden:

```bash
docker compose down
unset COMPOSE_PROJECT_NAME
```

## 8. Abschliessen und abgeben

- [ ] Eventuelle Korrekturen ins ursprüngliche Repository übernehmen.
- [ ] README mit tatsächlichen Ergebnissen und verbleibenden Einschränkungen abgleichen.
- [ ] Letzte Änderungen committen und pushen.
- [ ] Auf GitHub prüfen, dass Beispieldaten, Modell und alle Skripte vorhanden sind.
- [ ] Sicherstellen, dass der Dozent auf das Repository zugreifen kann.
- [ ] Repository-Link über Teams abgeben: **bis 15.11.2026, 23:59 Uhr**.

**Abgabebereit ist das Projekt nach erfolgreichem frischem Replay, Neutraining und dokumentierter Live-Inferenz sowie vollständiger Bereitstellung der Dateien und Dokumentation.**
