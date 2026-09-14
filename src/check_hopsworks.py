import os
from pathlib import Path

import hopsworks
from dotenv import load_dotenv


project_dir = Path(__file__).resolve().parents[1]
load_dotenv(project_dir / ".env", override=True)

required = [
    "HOPSWORKS_HOST",
    "HOPSWORKS_PROJECT",
    "HOPSWORKS_API_KEY",
]

for name in required:
    if not os.getenv(name):
        raise SystemExit(f"Fehlender Eintrag in .env: {name}")

project = hopsworks.login(
    host=os.environ["HOPSWORKS_HOST"],
    project=os.environ["HOPSWORKS_PROJECT"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
)

feature_store = project.get_feature_store()

print(f"Verbindung erfolgreich: {project.name}")
print(f"Featurestore: {feature_store.name}")
