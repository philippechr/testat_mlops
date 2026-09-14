Bis der Dozent die README vollständig nachvollziehen kann, bleiben diese To-dos:
	1	Genügend echte Daten sammeln.
        Fortschritt prüfen:docker compose exec python python src/feature_pipeline.py --prepare-only
        Benötigt werden mindestens 24 vollständige Trainingszeilen, nicht nur Rohbeobachtungen.

	2	Einen festen Beispieldatensatz erstellen.
        Collector stoppen und die gesammelten Dateien kopieren:
        docker compose stop python

	3	mkdir -p data/sample

	4	cp data/raw/*.json data/sample/

	5	Den README-Ablauf mit einem frischen Hopsworks-Projekt testen.
        Neues Projekt in .env eintragen und nacheinander ausführen:
        docker compose run --rm python python src/check_hopsworks.py

	6	docker compose run --rm python python src/feature_pipeline.py --upload-only --data-dir /app/data/sample

	7	docker compose run --rm python python src/training_pipeline.py

	8	docker compose run --rm python python src/inference_pipeline.py --replay --data-dir /app/data/sample
        Alle vier Befehle müssen erfolgreich durchlaufen.

	9	Ergebnisse kontrollieren.
        MLflow enthält einen erfolgreichen Trainingsrun mit Metriken und Artefakten. Unter models/ liegen latest.json und das zugehörige vollständige Run-Verzeichnis. Replay liefert eine Vorhersage.

	10	Eine echte Live-Inferenz erfolgreich ausführen.
        Collector wieder starten:
        docker compose up -d
        
        Sobald aktuelle Features vorhanden sind:
        docker compose exec python python src/inference_pipeline.py

	11	Alles für die Abgabe auf GitHub hochladen.
        Dazu gehören Skripte, Compose-Datei, .devcontainer/, .env.example, README, data/sample/, models/latest.json und das zugehörige Modellverzeichnis. Keine .env hochladen.
	
    12	Einen neuen Codespace erstellen und die README exakt durchspielen.Eigene Secrets hinterlegen, Beispieldaten importieren, trainieren und Replay ausführen. Damit prüfst du, dass keine lokalen Dateien oder Einstellungen fehlen.
	
    13	README abschliessend abgleichen und Repository-Link abgeben.Tatsächlich aufgetretene Probleme dokumentieren und sicherstellen, dass der Dozent auf das Repository zugreifen kann. Abgabe über Teams bis 15.11.2026, 23:59 Uhr.

Entscheidend ist Schritt 7: Erst der erfolgreiche Durchlauf aus einem frischen Codespace bestätigt, dass die Abgabe anhand der README reproduzierbar ist.