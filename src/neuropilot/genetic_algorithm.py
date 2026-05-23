"""Genetic algorithm trainer for WinnerMind Tennis Shot Strategy AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np

from neuropilot.config import GENETIC_CONFIG, MODELS_DIR, REPORTS_DIR, GeneticConfig
from neuropilot.model import build_pilot_model
from neuropilot.plotting import save_fitness_chart
from neuropilot.tennis_env import ShotResult, ShotScenario, decode_outputs, simulate_shot, standard_scenarios


@dataclass
class EvaluatedIndividual:
    weights: list[np.ndarray]
    fitness: float
    results: list[ShotResult]

    @property
    def best_result(self) -> ShotResult:
        return max(self.results, key=lambda result: result.fitness)


def evaluate_weights(weights: list[np.ndarray], scenarios: list[ShotScenario]) -> EvaluatedIndividual:
    model = build_pilot_model()
    model.set_weights(weights)
    results: list[ShotResult] = []

    for scenario in scenarios:
        outputs = model.predict(scenario.as_model_input(), verbose=0)[0]
        params = decode_outputs(outputs)
        results.append(simulate_shot(scenario, params))

    fitness = float(np.mean([result.fitness for result in results]))
    return EvaluatedIndividual(weights=weights, fitness=fitness, results=results)


def clone_weights(weights: list[np.ndarray]) -> list[np.ndarray]:
    return [layer.copy() for layer in weights]


def crossover(parent_a: list[np.ndarray], parent_b: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    child: list[np.ndarray] = []
    for weights_a, weights_b in zip(parent_a, parent_b):
        mask = rng.random(weights_a.shape) < 0.5
        child.append(np.where(mask, weights_a, weights_b))
    return child


def mutate(weights: list[np.ndarray], rng: np.random.Generator, config: GeneticConfig = GENETIC_CONFIG) -> list[np.ndarray]:
    mutated: list[np.ndarray] = []
    for layer in weights:
        mask = rng.random(layer.shape) < config.mutation_rate
        noise = rng.normal(0.0, config.mutation_strength, layer.shape)
        mutated.append(layer + mask * noise)
    return mutated


def write_history(rows: list[dict[str, float | int]], history_path: Path) -> None:
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["generation", "best_fitness", "average_fitness"])
        writer.writeheader()
        writer.writerows(rows)


def write_best_shot(result: ShotResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shot_json = json.dumps(result.to_visualization_dict(), indent=2)
    output_path.write_text(shot_json, encoding="utf-8")
    output_path.with_suffix(".js").write_text(
        f"window.WINNERMIND_BEST_SHOT = {shot_json};\n",
        encoding="utf-8",
    )


def train(
    config: GeneticConfig = GENETIC_CONFIG,
    model_path: Path = MODELS_DIR / "winnermind_best.weights.h5",
    history_path: Path = REPORTS_DIR / "training_history.csv",
    best_shot_path: Path = REPORTS_DIR / "best_shot.json",
) -> EvaluatedIndividual:
    """Train shot strategy with a genetic algorithm and save artifacts."""
    rng = np.random.default_rng(config.seed)
    scenarios = standard_scenarios()
    base_model = build_pilot_model()
    population = [clone_weights(base_model.get_weights()) for _ in range(config.population_size)]

    for individual in population:
        for layer in individual:
            layer += rng.normal(0.0, 1.0, layer.shape)

    best_history: list[float] = []
    average_history: list[float] = []
    rows: list[dict[str, float | int]] = []
    best: EvaluatedIndividual | None = None

    for generation in range(config.generations):
        evaluated = sorted(
            [evaluate_weights(weights, scenarios) for weights in population],
            key=lambda item: item.fitness,
            reverse=True,
        )
        best = evaluated[0]
        average = float(np.mean([item.fitness for item in evaluated]))
        best_history.append(best.fitness)
        average_history.append(average)
        rows.append({"generation": generation + 1, "best_fitness": best.fitness, "average_fitness": average})
        print(f"Generation {generation + 1:02d}: best={best.fitness:.2f}, average={average:.2f}")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        write_history(rows, history_path)
        save_fitness_chart(best_history, average_history, REPORTS_DIR / "fitness.png")
        write_best_shot(best.best_result, best_shot_path)

        elites = evaluated[: config.elite_count]
        next_population = [clone_weights(item.weights) for item in elites]
        while len(next_population) < config.population_size:
            parent_a, parent_b = rng.choice(elites, size=2, replace=True)
            child = crossover(parent_a.weights, parent_b.weights, rng)
            next_population.append(mutate(child, rng, config))
        population = next_population

    if best is None:
        raise RuntimeError("Training did not evaluate any individual.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    final_model = build_pilot_model()
    final_model.set_weights(best.weights)
    final_model.save_weights(model_path)

    write_history(rows, history_path)
    save_fitness_chart(best_history, average_history, REPORTS_DIR / "fitness.png")
    write_best_shot(best.best_result, best_shot_path)
    return best
