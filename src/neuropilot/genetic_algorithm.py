"""Genetic algorithm training utilities.

The complete implementation will evaluate many Keras models, select the best
weights, breed new candidates, mutate them, and track fitness improvement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FitnessResult:
    fitness: float
    landed: bool
    steps: int


def score_landing(final_x: float, final_y: float, final_vx: float, final_vy: float, landed: bool) -> float:
    """Initial scoring helper for the lander fitness function."""
    distance_penalty = abs(final_x - 50.0) * 2.0 + abs(final_y)
    speed_penalty = abs(final_vx) * 20.0 + abs(final_vy) * 25.0
    landing_bonus = 1000.0 if landed else 0.0
    return landing_bonus - distance_penalty - speed_penalty
