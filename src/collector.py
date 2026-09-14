"""Run the existing feature pipeline periodically while this container runs."""

import logging
import signal
import subprocess
import sys
import threading
from pathlib import Path

INTERVAL_SECONDS = 60 * 60
RETRY_SECONDS = 5 * 60
PIPELINE = Path(__file__).with_name("feature_pipeline.py")
stop = threading.Event()


def request_stop(signum, frame):
    stop.set()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    if not PIPELINE.is_file():
        raise SystemExit(f"Skript fehlt: {PIPELINE}")

    logging.info("Sammler gestartet: sofortiger Abruf, danach jede Stunde.")
    while not stop.is_set():
        try:
            result = subprocess.run(
                [sys.executable, "-u", str(PIPELINE)],
                timeout=900,
                check=False,
            )
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            logging.error("Abruf nach 900 Sekunden abgebrochen.")
            success = False

        if stop.is_set():
            break
        delay = INTERVAL_SECONDS if success else RETRY_SECONDS
        if success:
            logging.info("Abruf erfolgreich. Nächster Abruf in 60 Minuten.")
        else:
            logging.error("Abruf fehlgeschlagen. Neuer Versuch in 5 Minuten.")
        stop.wait(delay)

    logging.info("Sammler beendet.")


if __name__ == "__main__":
    main()
