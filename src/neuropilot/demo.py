"""Read the best trained WinnerMind shot data."""

from __future__ import annotations

import json

from neuropilot.config import REPORTS_DIR


def main() -> None:
    best_shot_path = REPORTS_DIR / "best_shot.json"
    if not best_shot_path.exists():
        raise FileNotFoundError("Train first: python -m neuropilot.train")

    data = json.loads(best_shot_path.read_text(encoding="utf-8"))
    print("Best trained shot:")
    print(json.dumps(data, indent=2))
    print("\nOpen visualization:")
    print("http://localhost:8000/reports/tennis_court_realistic_3d.html")


if __name__ == "__main__":
    main()
