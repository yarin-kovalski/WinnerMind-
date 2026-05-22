"""Shared configuration for the NeuroPilot training project."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    width: float = 100.0
    height: float = 100.0
    landing_pad_x: float = 50.0
    gravity: float = -0.08
    max_steps: int = 600
    initial_fuel: float = 100.0


@dataclass(frozen=True)
class GeneticConfig:
    population_size: int = 40
    elite_count: int = 6
    generations: int = 50
    mutation_rate: float = 0.08
    mutation_strength: float = 0.15


ENV_CONFIG = EnvironmentConfig()
GENETIC_CONFIG = GeneticConfig()
