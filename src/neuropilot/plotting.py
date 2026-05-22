"""Plotting helpers for training progress."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_fitness_chart(best_fitness: list[float], average_fitness: list[float], output_path: Path) -> None:
    """Save a chart showing how training fitness improves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label="Best fitness")
    plt.plot(average_fitness, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("NeuroPilot Genetic Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
