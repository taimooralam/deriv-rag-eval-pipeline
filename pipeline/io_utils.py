"""Tiny JSON IO helpers used across stages."""
import json
import os


def load_dotenv(path: str = ".env"):
    """Minimal stdlib .env loader: KEY=VALUE lines, '#' comments. Does not overwrite
    variables already present in the environment. Avoids a python-dotenv dependency."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
