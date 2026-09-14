# MLOps-Testat: Temperaturvorhersage für Windisch

Projektarbeit im CAS AI Operations an der FHNW. Ziel ist eine nachvollziehbare Feature–Training–Inference-Architektur (FTI) mit Hopsworks als Featurestore.

Vorhergesagt wird die Temperatur in Windisch, Schweiz, ungefähr drei Stunden nach einem Wetterbeobachtungszeitpunkt. MLflow dokumentiert Trainingsläufe. Modellperformance und freiwillige Ausbaustufen sind laut Aufgabenstellung keine Bewertungskriterien.

## 1. Stand und noch ausstehende Nachweise

| Bestandteil | Stand |
|---|---|
| Docker und Python-Laufzeit | Bereits erfolgreich gebaut und gestartet |
| OpenWeather-Anbindung und lokale Sammlung | Bereits erfolgreich ausgeführt |
| Hopsworks-Verbindung und Upload von Beobachtungen | Bereits erfolgreich ausgeführt |
| Collector mit 30-Minuten-Wartezeit | Code angepasst; nach Übernahme neu starten |
| Drei-Stunden-Features und verzögerte Labels | Implementiert; erfolgreicher Upload mit ausreichend echten Daten noch nachzuweisen |
| MLflow-Service | Image gebaut und Service gestartet; erster echter Trainingsrun noch ausstehend |
| Training-Pipeline | Implementiert; vollständiger Lauf mit ausreichenden Daten noch ausstehend |
| Live-Inferenz | Implementiert; vollständiger Lauf nach erfolgreichem Training noch ausstehend |
| Historisches Replay | Implementiert; vollständiger Lauf mit Modell und Beispieldatensatz noch ausstehend |
| Codespaces-Konfiguration | Bereitgestellt; frischer Codespace noch zu prüfen |
| Echter Beispieldatensatz und Abgabemodell | Noch zu erstellen und in Git aufzunehmen |

Dieser Stand beschreibt die Implementierung und die bisher nachgewiesenen Durchläufe getrennt. Vor Abgabe die Tabelle anhand der tatsächlichen Ergebnisse aktualisieren.

## 2. Architektur und Dateien

```text
OpenWeather
    |
    v
Collector -> Feature-Pipeline -> data/raw/
                    |
                    v
           Hopsworks Featurestore
                    |
                    v
            Training-Pipeline -> MLflow
                    |
                    v
              models/<run_id>/
                    |
                    v
            Inferenz-Pipeline
      Hopsworks-Aggregat + aktueller Temperaturwert
```

```text
.
├── compose.yaml
├── .env.example
├── .gitignore
├── README.md
├── .devcontainer/
│   ├── devcontainer.json
│   └── compose.codespaces.yaml
├── src/
│   ├── collector.py
│   ├── check_hopsworks.py
│   ├── feature_pipeline.py
│   ├── weather_features.py
│   ├── training_pipeline.py
│   └── inference_pipeline.py
├── data/
│   ├── raw/          # lokale Sammlung
│   ├── sample/       # später: fester echter Beispieldatensatz in Git
│   └── predictions/  # lokale Live- und Replay-Ergebnisse
└── models/           # später: Modell und Dokumentation eines erfolgreichen Runs
```

`weather_features.py` ist ein Hilfsmodul der Feature-Pipeline. `collector.py` startet diese periodisch. Die drei fachlichen Pipelines bleiben Feature, Training und Inferenz.

## 3. Laufzeit und Abhängigkeiten

Voraussetzungen für lokale Ausführung:

- Docker mit laufender Linux-Engine, z.B. Docker Desktop.
- Docker Compose ab Version 2.24.0.
- Internetzugang, OpenWeather API-Key und Hopsworks-Projekt mit API-Key.

Python muss lokal nicht zusätzlich installiert sein. Das Image nutzt Python 3.12 auf `linux/amd64`. Auf Apple Silicon wird es emuliert, weil die verwendete `hops-deltalake`-Version ein Linux-Binärpaket für x86-64 benötigt.

Die vollständigen direkten Paketversionen stehen im eingebetteten Dockerfile der `compose.yaml`. Wesentliche Versionen sind Hopsworks 5.0.7, MLflow 3.10.1, Pandas 2.2.3, NumPy 2.2.6 und Scikit-learn 1.6.1. OpenTelemetry ist zur Protobuf-Anforderung von Hopsworks passend festgelegt.

Hopsworks-Client und Plattformversion müssen kompatibel sein. Die Verbindung zum bisherigen Projekt wurde erfolgreich geprüft. Bei Verwendung einer anderen Plattformversion die Client-Kompatibilität prüfen.

Ein erfolgreicher Trainingslauf exportiert alle tatsächlich installierten Python-Paketversionen nach `requirements.freeze.txt`. Das Basisimage ist bisher nicht per Digest und der gesamte Build nicht durch einen transitiven Lockfile eingefroren.

## 4. Lokal einrichten

Eine `.env` im Repository-Ordner erstellen:

```bash
cp .env.example .env
```

Inhalt mit eigenen Zugangsdaten ergänzen:

```dotenv
OPENWEATHER_API_KEY=DEIN_OPENWEATHER_API_KEY
WEATHER_CITY=Windisch
WEATHER_COUNTRY=CH
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PROJECT=testat_fhnw_mlops
HOPSWORKS_API_KEY=DEIN_HOPSWORKS_API_KEY
```

Bei einem eigenen Hopsworks-Projekt Host und Projektnamen ersetzen. Die Skripte laden `.env` mit Vorrang vor vorhandenen Umgebungsvariablen. Deshalb gehört die Datei nicht in Git und wird im Codespace nicht benötigt.

Alle lokalen Docker-Befehle im Repository-Ordner ausführen:

```bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

- `python` startet den Collector.
- `mlflow` startet Tracking-API und Weboberfläche.
- Der Repository-Ordner ist nach `/app` eingebunden.
- Die lokale MLflow-Oberfläche ist unter http://localhost:5001 erreichbar.
- Im Python-Container verwendet der Client `http://mlflow:5000`.

Nach Paketänderungen ist ein Image-Build nötig. Python-Dateien sind durch den Bind-Mount sofort verfügbar; nach Änderung des Collectors den Prozess neu starten:

```bash
docker compose restart python
```

Bei jedem Neustart startet der Collector sofort einen Durchlauf.

## 5. Datenquelle und historische Sammlung

OpenWeather-Endpunkte:

- `/geo/1.0/direct`: übersetzt `Windisch,CH` in Koordinaten.
- `/data/2.5/weather`: liefert Wetterwerte für diesen Standort mit `units=metric`.

Gespeichert werden Ort, Land, Koordinaten, Wetterzeitpunkt, Temperatur in °C, Luftfeuchtigkeit in %, Luftdruck in hPa, Windgeschwindigkeit in m/s und Bewölkung in %.

`observed_at` ist der Wetterzeitpunkt aus der API, nicht einfach die lokale Abrufzeit. Zeitpunkte werden in UTC verarbeitet: in JSON als ISO-Zeitstempel, in Hopsworks als Unix-Millisekunden.

Die historischen Daten entstehen durch das Sammeln aktueller Beobachtungen. Es werden keine künstlichen Wetterdaten erzeugt und keine rückwirkenden OpenWeather-History-Abfragen durchgeführt.

### Collector

Der Collector startet sofort, wartet nach erfolgreichem Abschluss 30 Minuten und führt dann die Feature-Pipeline erneut aus. Bei Fehlern folgt nach fünf Minuten ein neuer Versuch. Ein Durchlauf darf maximal 900 Sekunden dauern.

Die Pipeline-Laufzeit kommt zur Wartezeit hinzu. Es handelt sich nicht um einen festen Zeitplan zu jeder vollen und halben Stunde und nicht um eine strikte Begrenzung der API-Anfragen.

Ein erfolgreicher Durchlauf verursacht zwei OpenWeather-Anfragen: Ortssuche und Wetterabruf. Im Normalbetrieb sind das ungefähr vier Anfragen pro Stunde; Neustarts und Fehlerwiederholungen können zusätzliche Anfragen verursachen.

Der Wechsel von 60 auf 30 Minuten verhindert, dass bereits kleine Laufzeit- oder Zeitstempelverschiebungen regelmässig ein ganzes Stundenfenster der Drei-Stunden-Aggregation leer lassen. Er garantiert bei Ausfällen trotzdem keine vollständige Historie.

Docker, Rechner und Internet müssen verfügbar sein. Während des Ruhezustands oder bei ausgeschaltetem Rechner entstehen Lücken. Diese werden nicht automatisch nachgefüllt.

Logs anzeigen:

```bash
docker compose logs -f --tail=30 python
```

`Ctrl+C` beendet nur die Log-Anzeige. Die Dateien liegen lokal in `data/raw/`. Gleicher Standort und Wetterzeitpunkt ergeben denselben Dateinamen, sodass identische Zeitpunkte nicht als zusätzliche Beobachtungen gezählt werden.

## 6. Feature-Pipeline und Aggregation

Die Feature-Pipeline ruft Wetterdaten ab, speichert sie lokal, liest vorhandene JSON-Dateien, prüft Datentypen und erstellt Features sowie verfügbare Labels. Anschliessend schreibt sie die Dataframes nach Hopsworks.

Die Modell-Eingaben sind in `weather_features.py` in dieser Reihenfolge festgelegt:

```python
MODEL_FEATURES = ["temperature_mean_3h", "temperature_current"]
```

### Aggregiertes Feature

Für den Zeitpunkt `t` verwendet `temperature_mean_3h` ausschliesslich Beobachtungen in `[t-3h, t)`. Der Wert zum Zeitpunkt `t` geht nicht in die Aggregation ein.

Das Intervall wird in drei einstündige Abschnitte unterteilt. Pro Abschnitt wird ein Temperaturmittelwert berechnet, danach der Mittelwert dieser drei Werte. Mehrere Abrufe innerhalb einer Stunde geben dieser Stunde dadurch kein zusätzliches Gewicht.

Eine Feature-Zeile ist nur gültig, wenn:

- die Historie des Standorts mindestens bis `t-3h` zurückreicht;
- jeder der drei Stundenabschnitte mindestens eine echte Beobachtung enthält;
- keine Lücke zwischen Beobachtungen oder zu den Fensterrändern grösser als zwei Stunden ist.

Keine Interpolation und kein Auffüllen fehlender Werte. Der Mittelwert ist eine Schätzung aus echten Stichproben, kein kontinuierlich gemessener Temperaturmittelwert. Verwendete Anzahl und Zeitspanne werden als Metadaten gespeichert.

### Aktuelles Feature

Beim Training ist `temperature_current` die tatsächlich beobachtete Temperatur zum damaligen Zeitpunkt `t`. Bei der Live-Inferenz wird der entsprechende aktuelle Wert frisch von OpenWeather abgerufen. Eine neue HTTP-Anfrage garantiert keinen neuen Messzeitpunkt, weshalb dessen Alter zusätzlich geprüft wird.

### Label

`temperature_target_3h` ist eine echte Temperaturbeobachtung ungefähr drei Stunden später. Aus dem Fenster `t+3h ±30min` wird die zeitlich nächste Beobachtung gewählt; bei gleichem Abstand die frühere.

Erst wenn eine Beobachtung bei oder nach `t+3h30min` vorliegt, ist das Label-Zeitfenster abgeschlossen. Ohne passenden Wert gibt es keine Trainingszeile. Der tatsächliche Label-Zeitpunkt und seine Abweichung werden gespeichert, aber nicht als Modell-Eingaben verwendet.

Frühestens nach drei Stunden kann ein Feature entstehen, frühestens nach etwa 6,5 Stunden eine vollständige Trainingszeile. Ob es tatsächlich soweit ist, hängt von den Messzeitpunkten und Lücken ab.

### Modi

Nur lokal berechnen und anzeigen, ohne API-Aufrufe:

```bash
docker compose exec python python src/feature_pipeline.py --prepare-only
```

Vorhandene Daten nach Hopsworks laden, ohne neuen Wetterabruf:

```bash
docker compose exec python python src/feature_pipeline.py --upload-only
```

Beispieldaten importieren:

```bash
docker compose exec python python src/feature_pipeline.py --upload-only --data-dir /app/data/sample
```

Manuelle Uploads nicht parallel zum Collector starten. Bei Bedarf zuerst `docker compose stop python` verwenden und den Befehl mit `docker compose run --rm python ...` einmalig ausführen. Anschliessend `docker compose up -d`.

## 7. Hopsworks

| Feature Group | Version | Inhalt |
|---|---:|---|
| `weather_observations` | 1 | Wetterbeobachtungen |
| `weather_features` | 2 | Gültige Drei-Stunden-Features, auch ohne zukünftiges Label |
| `weather_training_features` | 2 | Dieselben Features mit vorhandenem echtem Label |

Schlüssel: `location_id` und `observed_at`. Event Time: `observed_at`. Alle drei Gruppen verwenden Upserts. Neue Gruppen entstehen erst bei einem nichtleeren Upload.

Version 2 trennt die Drei-Stunden-Definition von der zuvor geplanten 24-Stunden-Definition. Die Änderung des Sammelintervalls ändert die Feature-Definition nicht; deshalb bleibt Version 2 bestehen.

Die bewusste Kopie der Features in die Trainingsgruppe hält den Trainingsabruf einfach: Eine Feature View kann Features und Label ohne Join auswählen. Aktuelle Features bleiben unabhängig von noch unbekannten Labels verfügbar.

Beim kleinen Projektdatensatz werden alle lokalen Beobachtungen erneut verarbeitet und hochgeladen. Ein inkrementeller Upload-Checkpoint ist noch nicht implementiert.

Verbindung prüfen:

```bash
docker compose exec python python src/check_hopsworks.py
```

## 8. Training und Evaluation

Training wird manuell gestartet; der Collector trainiert kein Modell:

```bash
docker compose exec python python src/training_pipeline.py
```

Die Pipeline:

1. Prüft MLflow und lädt die Trainings-Feature-Group Version 2.
2. Erstellt oder lädt `weather_temperature_training`, Feature-View-Version 2, mit einer Query für Features, Label und benötigte Metadaten.
3. Erstellt einen versionierten Parquet-Trainingsdatensatz aus der View und lädt ihn über die View.
4. Verwendet ungefähr die älteren 80 % der Zeilen für Training und die neueren 20 % für Test.
5. Entfernt Trainingszeilen am Übergang, deren Label-Zeitfenster erst während des Testzeitraums endet.
6. Trainiert `StandardScaler -> Ridge(alpha=1.0)`. Der Scaler wird nur auf dem Training angepasst.
7. Bewertet Modell und Persistenz-Referenz auf denselben Testzeilen.
8. Speichert Tracking-Ergebnisse und exportiert das evaluierte Modell.

Die Referenz sagt voraus: Temperatur in drei Stunden = aktuelle Temperatur.

Metriken sind MAE und RMSE, beide in °C. Niedrigere Werte bedeuten kleinere Fehler. Auf dem Testdatensatz wird nicht nachtrainiert. Nur die beiden festgelegten Features gelangen ins Modell; spätere Label-Zeitpunkte sind keine Eingaben.

Technische Mindestwerte: 24 vollständige Zeilen insgesamt, mindestens zwölf Trainings- und fünf Testzeilen nach dem zeitlichen Ausschluss. Das sind Projektentscheidungen für einen Demonstrationslauf, keine Vorgaben der Aufgabenstellung und keine statistische Qualitätsgarantie.

Fehlen Daten, endet das Skript mit Exit-Code 2 und einer Erklärung. Es entsteht kein vorgetäuschter Trainingsrun. Optional kann `--alpha` geändert werden; bei mehreren Standorten verlangt das Skript eine Auswahl mit `--location-id`.

## 9. MLflow und Modelle

Experiment: `windisch_temperature_3h`.

Pro Trainingsrun werden Modellparameter, Feature-Reihenfolge, Zeiträume, Zeilenzahlen, Hopsworks-Versionen, MAE/RMSE von Modell und Referenz sowie Modellartefakte protokolliert.

Die Artefakte enthalten echte Datensnapshots, Vorhersagen, Vergleichsgrafik, Paketversionen, verwendete Skripte, Modell und Metadaten. Das Trainingsskript prüft nach dem lokalen Speichern, ob das geladene Modell dieselben Testvorhersagen liefert.

MLflow nutzt eine SQLite-Datenbank und Artefaktspeicher im benannten Docker-Volume `mlflow-data`. Normale Neustarts erhalten dieses Volume. Es wird nicht automatisch nach GitHub oder in einen neuen Codespace übertragen.

Nach erfolgreichem Training:

```text
models/
├── latest.json
└── <run_id>/
    ├── model.joblib
    ├── metadata.json
    ├── dataset.csv
    ├── train.csv
    ├── test.csv
    ├── predictions.csv
    ├── evaluation.png
    ├── requirements.freeze.txt
    ├── training_pipeline.py
    └── weather_features.py
```

`latest.json` verweist erst nach vollständig erfolgreichem Tracking auf den neuen Run. Die Inferenz lädt das Modell lokal über diesen Verweis. Eine zusätzliche Model Registry ist nicht implementiert; das lokale Speichern ist auf Seite 1 der Aufgabenstellung ausdrücklich zulässig. MLflow-Tracking und Modellartefakte werden nicht als Model Registry bezeichnet.

## 10. Live-Inferenz und Aktualität

```bash
docker compose exec python python src/inference_pipeline.py
```

Ablauf:

1. Modell und Metadaten über `models/latest.json` laden.
2. Aktuelle Wetterbeobachtung für Windisch frisch abrufen.
3. Über die Feature View `weather_temperature_inference`, Version 2, das neueste passende Aggregat aus `weather_features`, Version 2, laden.
4. Standort, Eingabeschema und Zeitpunkte prüfen.
5. Aggregat und aktuellen Temperaturwert gemeinsam an `model.predict()` übergeben.
6. Ergebnis und verwendete Eingaben als JSON unter `data/predictions/` speichern.

Zeitliche Regeln:

- Die Wetterbeobachtung darf zum Ausführungszeitpunkt höchstens 30 Minuten alt sein.
- Das Aggregat darf nicht aus der Zukunft relativ zur Wetterbeobachtung stammen.
- Sein Bezugszeitpunkt darf maximal 90 Minuten vor der Wetterbeobachtung liegen.
- Der Zielzeitpunkt ist Wetterbeobachtungszeitpunkt + drei Stunden, nicht exakt Rechnerzeit + drei Stunden.

Das Aggregat bezieht sich auf die drei Stunden vor seinem eigenen Berechnungszeitpunkt. Es ist bei Live-Inferenz nicht zwingend der exakte Durchschnitt der letzten drei Stunden vor der frischen Wetterbeobachtung. Diese Verzögerung ist eine bewusste Einschränkung der Batch-/RT-Kombination. Beim Training sind beide Features zeitlich auf dieselbe historische Beobachtung ausgerichtet; bei Live-Inferenz kann es dadurch eine Abweichung zwischen Training und Anwendung geben.

Ausgabe und JSON dokumentieren beide Zeitpunkte und `aggregate_lag_minutes`. Die 90-Minuten-Grenze begrenzt die Abweichung, beseitigt sie aber nicht. Bei zu alten oder fehlenden Features bricht die Inferenz ab, statt unbemerkt veraltete Werte zu verwenden.

Kein Modell vorhanden ist in der aktuellen Sammelphase ein erwarteter Abbruch, kein erfolgreicher Inferenznachweis.

## 11. Historisches Replay für die Abgabe

Alte Beispieldaten erfüllen die Aktualitätsprüfung der Live-Inferenz nicht. Deshalb gibt es einen ausdrücklich getrennten Replay-Modus:

```bash
docker compose exec python python src/inference_pipeline.py --replay --data-dir /app/data/sample
```

Dieser Modus:

- verwendet eine echte gespeicherte Beobachtung als damaligen aktuellen Temperaturwert;
- führt keinen OpenWeather-Aufruf durch;
- wählt standardmässig die neueste passende Beobachtung im Beispieldatensatz;
- verlangt einen Zeitpunkt nach allen im Modelltraining verwendeten Labels;
- lädt das zu diesem historischen Zeitpunkt passende Aggregat über Hopsworks;
- behält die 90-Minuten-Grenze relativ zum historischen Wetterzeitpunkt bei;
- kennzeichnet Ausgabe und JSON als `replay`.

Die Prüfung des Wetteralters gegen die heutige Rechnerzeit wird nur im Replay ausgesetzt. Die historische Beobachtung wird nicht als heutige Live-Messung ausgegeben. Es werden keine künstlichen Temperaturen erzeugt. Replay ersetzt nicht den zusätzlich ausstehenden Nachweis einer echten Live-Inferenz.

Mit `--at` kann ein exakter vorhandener ISO-Wetterzeitpunkt mit Zeitzone ausgewählt werden. Er muss nach dem Trainingsende liegen. Ohne passende Beobachtung erfolgt ein verständlicher Abbruch.

Vor Replay müssen die Beispieldaten in Hopsworks importiert worden sein und ein passendes Modell samt Metadaten vorliegen. Der Replay-Modus benötigt weiterhin Internet und Hopsworks-Zugang.

## 12. Echten Beispieldatensatz und Abgabemodell vorbereiten

Erst durchführen, wenn genügend echte Beobachtungen für Training und spätere Replay-Zeitpunkte vorhanden sind. Die folgenden Befehle werden lokal im Repository ausgeführt:

```bash
docker compose stop python
mkdir -p data/sample
cp data/raw/*.json data/sample/
docker compose run --rm python python src/feature_pipeline.py --prepare-only --data-dir /app/data/sample
docker compose run --rm python python src/feature_pipeline.py --upload-only --data-dir /app/data/sample
docker compose run --rm python python src/training_pipeline.py
docker compose run --rm python python src/inference_pipeline.py --replay --data-dir /app/data/sample
docker compose up -d
```

Die Kopie fixiert einen echten Datenstand; nichts wird generiert. Beim erstmaligen Erstellen ist `data/sample/` leer. Spätere Kopien ergänzen/überschreiben Dateien, entfernen aber keine alten Dateien. Für eine gezielte Neuauswahl den Inhalt bewusst überprüfen.

Danach `data/sample/`, `models/latest.json` und das vollständige zugehörige Run-Verzeichnis mit Git aufnehmen. Vorhersageausgaben unter `data/predictions/` bleiben lokal; einen erfolgreichen Lauf beispielsweise im README dokumentieren.

Wichtig: Die Training-Pipeline liest die Daten aus Hopsworks. Ein bestehendes Projekt kann zusätzliche historische Daten enthalten. Für einen eindeutig isolierten Abgabenachweis ein frisches Hopsworks-Projekt mit ausschliesslich den Beispieldaten verwenden. Der tatsächlich verwendete Datensnapshot und seine Prüfsumme werden im Modell-Run gespeichert.

Der Beispieldatensatz muss auch Beobachtungen nach dem Trainingsende des bereitgestellten Modells enthalten, damit Replay möglich ist. Ein Snapshot ohne solche Beobachtungen reicht nicht.

## 13. GitHub Codespaces

Die Konfiguration verwendet das bestehende Compose-Image und startet Python sowie MLflow. Im Codespace läuft kein automatischer Collector; die Pipelines werden einzeln im Terminal ausgeführt. Das Terminal befindet sich bereits im Python-Container unter `/app`.

### Zugangsdaten

Vor Erstellung des Codespaces folgende GitHub-Codespaces-Secrets für das Repository bereitstellen:

- `HOPSWORKS_HOST`
- `HOPSWORKS_PROJECT`
- `HOPSWORKS_API_KEY`
- `OPENWEATHER_API_KEY` für Live-Abrufe

Die Devcontainer-Konfiguration übergibt sie an die Terminalumgebung. In Codespaces keine `.env` mit leeren oder alten Werten anlegen: Die Skripte würden diese bevorzugen. Für ein reines Replay mit bereitgestellten Beispieldaten ist kein OpenWeather-Key erforderlich.

### Ablauf im Codespace-Terminal

```bash
python src/check_hopsworks.py
python src/feature_pipeline.py --prepare-only --data-dir /app/data/sample
python src/feature_pipeline.py --upload-only --data-dir /app/data/sample
python src/training_pipeline.py
python src/inference_pipeline.py --replay --data-dir /app/data/sample
```

Alternativ kann nach dem Datenimport das bereitgestellte Modell unmittelbar für Replay verwendet werden. Das Ausführen der Training-Pipeline bleibt Bestandteil des vollständigen Nachweises.

MLflow wird über den weitergeleiteten Port `mlflow:5000` geöffnet. In der Ports-Ansicht den Eintrag MLflow auswählen und die Sichtbarkeit privat lassen. Die lokale Browseradresse `localhost:5001` gilt für Docker auf dem eigenen Rechner, nicht für den Browserzugriff auf Codespaces.

Für Live-Inferenz braucht der Featurestore ausreichend aktuelle Beobachtungen und Aggregate. Historische Beispieldaten allein reichen dafür nicht. Wenn im eigenen Hopsworks-Projekt noch keine aktuelle Historie existiert, kann `python src/collector.py` bewusst im Codespace-Terminal gestartet werden; der Codespace muss dafür laufen. Nach ausreichender Sammlung Live-Inferenz in einem zweiten Terminal ausführen. Keine dauerhafte Sammlung über Inaktivitätsstopps hinweg erwarten.

Ein neuer Codespace beginnt mit eigenem, zunächst leerem MLflow-Speicher. Das bereitgestellte Modell und neu ausführbares Training machen die Abgabe unabhängig von der lokalen MLflow-Historie.

Die Konfiguration muss vor Abgabe einmal in einem frischen Codespace vollständig getestet werden.

## 14. Git, Nachweise und Abgabe

`.gitignore` schliesst `.env`, Python-Zwischendateien, `data/raw/` und `data/predictions/` aus. `data/sample/` und `models/` werden bewusst nicht ausgeschlossen. Leere Ordner werden von Git nicht gespeichert.

Vor Abgabe nachzuweisen:

- Feature-, Trainings- und Inferenzskripte sind im GitHub-Repository vorhanden.
- Echte aggregierte Features und Labels wurden in Hopsworks gespeichert.
- Ein Modell wurde aus einem Feature-View-Trainingsdatensatz trainiert und evaluiert.
- Das exportierte Modell und seine Metadaten sind vorhanden.
- Eine Live-Inferenz hat beide Features tatsächlich verwendet.
- Replay und/oder vollständiges Neutraining funktionieren mit dem bereitgestellten echten Beispieldatensatz.
- Einrichtung und Ausführung wurden in einer frischen Umgebung geprüft.
- README-Status, tatsächlicher Trainingszeitraum, Zeilenzahlen, Run-ID und Metriken sind aktualisiert.
- Verbleibende Probleme sind ausdrücklich beschrieben.

Abgabe: Link zum GitHub-Repository über die Teams-Aufgabe. Termin laut Aufgabenstellung: **15.11.2026, 23:59 Uhr**.

## 15. Bezug zur Bewertung und Einschränkungen

| Kriterium | Umsetzung |
|---|---|
| Strukturiertes Vorgehen | Getrennte FTI-Skripte; Features werden tatsächlich in Hopsworks gespeichert |
| Historische und aktuelle Daten | Historische Sammlung fürs Training; frischer API-Abruf bei Live-Inferenz; getrennter historischer Replay-Modus |
| Aggregiertes Feature | Drei-Stunden-Mittelwert aus mehreren Zeitpunkten |
| RT-Feature | Aktuelle Temperatur beim Inferenzaufruf |
| Vollständigkeit | Alle Schritte implementiert; erfolgreiche Durchläufe teilweise noch nachzuweisen |
| Dokumentation | Daten, Features, Modell, Abhängigkeiten, Befehle und Bereitstellung beschrieben |
| Reflexion | Einschränkungen und noch nicht geprüfte Teile offengelegt |

Das 24-Stunden-Luftfeuchtigkeitsfeature und Bewölkung in der Aufgabenstellung sind Beispiele, keine verpflichtenden Variablen oder Zeitfenster. Eine Model Registry ist aufgrund der ausdrücklichen Erlaubnis zur lokalen Modellspeicherung nicht erforderlich. Docker und MLflow sind freiwillige Erweiterungen. Modellperformance, Codequalität und Ausbaustufen werden laut Aufgabenstellung nicht bewertet.

Verbleibende Einschränkungen:

- Ein Standort und anfangs ein kurzer Zeitraum; keine ausreichende Abdeckung von Jahreszeiten oder Wetterlagen.
- Kleine, zeitlich abhängige Trainings- und Testdaten; Metriken sind zunächst nur Demonstrationsergebnisse.
- Rechner-, Netzwerk- und API-Ausfälle erzeugen Datenlücken.
- Wechsel auf 30 Minuten verbessert die Datenabdeckung, garantiert sie aber nicht.
- Labels liegen innerhalb einer Toleranz von ±30 Minuten um den Zielzeitpunkt.
- Batch-Aggregat und RT-Temperatur können bei Live-Inferenz unterschiedliche Bezugszeitpunkte haben.
- Wiederholte Auswahl von Modellparametern anhand desselben Tests würde dessen Unabhängigkeit beeinträchtigen.
- Vollständige Neu-Uploads aller lokalen Daten sind für grosse Datenmengen ineffizient.
- Pakete und Basisimage sind noch nicht über alle Ebenen vollständig eingefroren.
- Codespaces und vollständige Trainings-/Inferenzläufe müssen noch mit echten Daten verifiziert werden.

## Quellen

- Aufgabenstellung „Projektarbeit MLOps“, FHNW.
- OpenWeather: https://openweathermap.org/api
- Hopsworks Feature Groups: https://docs.hopsworks.ai/latest/python-api/hsfs/feature_group/
- Hopsworks Feature Views: https://docs.hopsworks.ai/latest/python-api/hsfs/feature_view/
- MLflow: https://mlflow.org/docs/latest/
- Devcontainer-Spezifikation: https://containers.dev/implementors/json_reference/
- Codespaces-Secrets: https://docs.github.com/en/codespaces/managing-your-codespaces/managing-your-account-specific-secrets-for-github-codespaces
