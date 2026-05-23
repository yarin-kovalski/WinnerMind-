"""Shared configuration for AceMind Tennis Shot Strategy AI."""

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


@dataclass(frozen=True)
class CourtConfig:
    court_length: float = 23.77
    doubles_width: float = 10.97
    singles_width: float = 8.23
    run_back: float = 8.23
    side_run: float = 4.57
    net_center_height: float = 0.914
    net_post_height: float = 1.07
    contact_height: float = 0.85
    min_power: float = 0.45
    max_power: float = 1.0
    min_launch_angle: float = 8.0
    max_launch_angle: float = 28.0


@dataclass(frozen=True)
class GeneticConfig:
    population_size: int = 44
    elite_count: int = 7
    generations: int = 40
    mutation_rate: float = 0.08
    mutation_strength: float = 0.16
    seed: int = 7


COURT_CONFIG = CourtConfig()
GENETIC_CONFIG = GeneticConfig()
