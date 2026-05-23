"""Run a trained WinnerMind model on a fresh tennis situation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuropilot.config import MODELS_DIR, REPORTS_DIR
from neuropilot.genetic_algorithm import write_best_shot
from neuropilot.model import build_pilot_model
from neuropilot.tennis_env import ShotResult, decode_outputs, random_scenario, simulate_shot


WEIGHTS_PATH = MODELS_DIR / "winnermind_best.weights.h5"


def predict_random_shot(seed: int | None = None, weights_path: Path = WEIGHTS_PATH) -> ShotResult:
    """Load trained weights, create a random incoming ball, and return the model's chosen shot."""
    if not weights_path.exists():
        raise FileNotFoundError("Missing trained weights. Run: python -m neuropilot.train")

    model = build_pilot_model()
    model.load_weights(weights_path)

    scenario = random_scenario(seed=seed)
    outputs = model.predict(scenario.as_model_input(), verbose=0)[0]
    params = decode_outputs(outputs)
    return simulate_shot(scenario, params)


def print_assignment_map() -> None:
    print("Assignment 3 Exercise 3 checklist:")
    print("- Keras neural network: src/neuropilot/model.py")
    print("- Genetic algorithm: src/neuropilot/genetic_algorithm.py")
    print("- Fitness function: src/neuropilot/tennis_env.py")
    print("- Fitness progress chart: reports/fitness.png")
    print("- Saved model weights: models/winnermind_best.weights.h5")
    print("- GUI option: python -m neuropilot.gui")
    print("- 3D wow visualization: reports/tennis_court_realistic_3d.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WinnerMind on a random incoming tennis ball.")
    parser.add_argument("--seed", type=int, default=None, help="Use a fixed seed for repeatable demo shots.")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Only print the shot; do not update reports/best_shot.json and reports/best_shot.js.",
    )
    args = parser.parse_args()

    result = predict_random_shot(seed=args.seed)
    data = result.to_visualization_dict()

    if not args.no_export:
        write_best_shot(result, REPORTS_DIR / "best_shot.json")

    print_assignment_map()
    print("\nRandom incoming ball:")
    print(f"- x: {data['incomingX']}m")
    print(f"- z: {data['incomingZ']}m")
    print(f"- height: {data['incomingHeight']}m")
    print(f"- speed: {data['incomingSpeed']}m/s")
    print(f"- spin: {data['incomingSpin']}")

    print("\nWinnerMind chosen return:")
    print(f"- power: {data['power']}")
    print(f"- launch angle: {data['launchAngleDeg']} deg")
    print(f"- arc height: {data['arcHeight']}m")
    print(f"- spin: {data['spin']} ({'topspin' if data['spin'] >= 0 else 'slice/backspin'})")
    print(f"- target: x={data['targetX']}m, z={data['targetZ']}m")
    print(f"- clears net: {data['clearedNet']}")
    print(f"- in court: {data['inCourt']}")
    print(f"- fitness: {data['fitness']}")

    if not args.no_export:
        print("\nUpdated visualization data:")
        print("- reports/best_shot.json")
        print("- reports/best_shot.js")

    print("\nOpen visualization:")
    print("file:///C:/Users/ASUS-H170M/Desktop/from_idea/exercsie3/reports/tennis_court_realistic_3d.html")
    print("\nFull shot JSON:")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
