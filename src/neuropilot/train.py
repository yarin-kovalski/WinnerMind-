"""Training entry point for WinnerMind Tennis Shot Strategy AI."""

from __future__ import annotations

from neuropilot.genetic_algorithm import train


def main() -> None:
    best = train()
    best_shot = best.best_result
    print(f"\nTraining complete. Best average fitness: {best.fitness:.2f}")
    print(f"Best visual shot fitness: {best_shot.fitness:.2f}")
    print("Saved weights: models/winnermind_best.weights.h5")
    print("Saved history: reports/training_history.csv")
    print("Saved chart: reports/fitness.png")
    print("Saved 3D animation data: reports/best_shot.json")
    print("Open visualization: http://localhost:8000/reports/tennis_court_realistic_3d.html")


if __name__ == "__main__":
    main()
