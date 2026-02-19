import json
import os

FILE = "session_history.json"


def load_history():
    if not os.path.exists(FILE):
        return []

    with open(FILE, "r") as f:
        return json.load(f)


def save_session(scores):
    history = load_history()
    history.append(scores)

    with open(FILE, "w") as f:
        json.dump(history, f)
