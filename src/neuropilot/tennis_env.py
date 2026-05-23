"""Tennis shot simulation and fitness scoring for WinnerMind."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sin, pi

import numpy as np

from neuropilot.config import COURT_CONFIG, CourtConfig


@dataclass(frozen=True)
class ShotScenario:
    incoming_x: float
    incoming_z: float
    incoming_height: float
    incoming_speed: float
    incoming_spin: float
    opponent_x: float
    opponent_z: float
    desired_target_x: float
    desired_target_z: float
    aggression: float

    def as_model_input(self, config: CourtConfig = COURT_CONFIG) -> np.ndarray:
        half_width = config.singles_width / 2
        half_length = config.court_length / 2
        return np.array(
            [
                [
                    (self.incoming_x + half_width) / config.singles_width,
                    self.incoming_z / half_length,
                    self.incoming_height / 3.5,
                    self.incoming_speed / 40.0,
                    (self.incoming_spin + 1.0) / 2.0,
                    (self.opponent_x + half_width) / config.singles_width,
                    (-self.opponent_z) / half_length,
                    self.aggression,
                ]
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class ShotParameters:
    power: float
    launch_angle_deg: float
    arc_height: float
    spin: float
    target_x: float
    target_z: float


@dataclass(frozen=True)
class ShotResult:
    scenario: ShotScenario
    parameters: ShotParameters
    trajectory: list[tuple[float, float, float]]
    net_height_at_crossing: float
    cleared_net: bool
    in_court: bool
    target_error: float
    opponent_distance: float
    fitness: float

    def to_visualization_dict(self) -> dict[str, float | int | str | list[list[float]]]:
        return {
            "strategy": "Aggressive Wide" if self.scenario.aggression > 0.65 else "Safe Placement",
            "power": round(self.parameters.power, 3),
            "launchAngleDeg": round(self.parameters.launch_angle_deg, 2),
            "arcHeight": round(self.parameters.arc_height, 3),
            "spin": round(self.parameters.spin, 3),
            "targetX": round(self.parameters.target_x, 3),
            "targetZ": round(self.parameters.target_z, 3),
            "fitness": round(self.fitness, 2),
            "clearedNet": self.cleared_net,
            "inCourt": self.in_court,
            "targetError": round(self.target_error, 3),
            "opponentDistance": round(self.opponent_distance, 3),
            "incomingX": round(self.scenario.incoming_x, 3),
            "incomingZ": round(self.scenario.incoming_z, 3),
            "incomingHeight": round(self.scenario.incoming_height, 3),
            "incomingSpeed": round(self.scenario.incoming_speed, 3),
            "incomingSpin": round(self.scenario.incoming_spin, 3),
            "trajectory": [[round(x, 3), round(y, 3), round(z, 3)] for x, y, z in self.trajectory],
        }


def decode_outputs(outputs: np.ndarray, config: CourtConfig = COURT_CONFIG) -> ShotParameters:
    power01, launch01, arc01, spin01, target_x01, target_z01 = [float(value) for value in outputs]
    half_width = config.singles_width / 2
    half_length = config.court_length / 2
    power = config.min_power + power01 * (config.max_power - config.min_power)
    launch = config.min_launch_angle + launch01 * (config.max_launch_angle - config.min_launch_angle)
    arc_height = 0.8 + arc01 * 2.6
    spin = spin01 * 2.0 - 1.0
    target_x = -half_width + target_x01 * config.singles_width
    target_z = -half_length + 1.0 + target_z01 * (half_length - 1.6)
    return ShotParameters(
        power=power,
        launch_angle_deg=launch,
        arc_height=arc_height,
        spin=spin,
        target_x=target_x,
        target_z=target_z,
    )


def net_height_at_x(x: float, config: CourtConfig = COURT_CONFIG) -> float:
    """Approximate the tennis net height at a horizontal crossing position."""
    half_net_width = config.doubles_width / 2 + 0.914
    ratio = min(1.0, abs(x) / half_net_width)
    return config.net_center_height + ratio * (config.net_post_height - config.net_center_height)


def simulate_shot(
    scenario: ShotScenario,
    params: ShotParameters,
    config: CourtConfig = COURT_CONFIG,
    steps: int = 72,
) -> ShotResult:
    """Simulate a tactical tennis shot and calculate fitness."""
    start_x = scenario.incoming_x
    start_z = scenario.incoming_z
    end_x = params.target_x
    end_z = params.target_z
    half_width = config.singles_width / 2
    half_length = config.court_length / 2

    launch_factor = (params.launch_angle_deg - config.min_launch_angle) / (
        config.max_launch_angle - config.min_launch_angle
    )
    topspin = max(0.0, params.spin)
    slice_spin = max(0.0, -params.spin)
    arc_height = params.arc_height + launch_factor * 0.45 + params.power * 0.35 - topspin * 0.25 + slice_spin * 0.18
    arc_height = max(0.45, arc_height)

    trajectory: list[tuple[float, float, float]] = []
    net_y = -999.0
    net_crossing_x = 0.0
    previous_x = start_x
    previous_y = config.contact_height
    previous_z = start_z

    for index in range(steps + 1):
        t = index / steps
        curve = (t * (1 - t)) * params.spin * 0.75
        x = start_x + (end_x - start_x) * t + curve
        z = start_z + (end_z - start_z) * t
        y = config.contact_height * (1 - t) + 0.08 * t + sin(pi * t) * arc_height
        trajectory.append((x, y, z))

        if (previous_z >= 0 >= z) or (previous_z <= 0 <= z):
            ratio = abs(previous_z) / max(0.0001, abs(previous_z - z))
            net_crossing_x = previous_x + ratio * (x - previous_x)
            net_y = previous_y + ratio * (y - previous_y)

        previous_x, previous_y, previous_z = x, y, z

    net_required_height = net_height_at_x(net_crossing_x, config)
    cleared_net = net_y > net_required_height + 0.08
    in_court = -half_width <= end_x <= half_width and -half_length <= end_z <= 0
    target_error = float(np.hypot(end_x - scenario.desired_target_x, end_z - scenario.desired_target_z))
    opponent_distance = float(np.hypot(end_x - scenario.opponent_x, end_z - scenario.opponent_z))

    fitness = 150.0
    speed_match = 1.0 - min(1.0, abs(params.power - scenario.incoming_speed / 40.0))
    fitness += params.power * 95.0 * scenario.aggression
    fitness += speed_match * 28.0
    fitness += abs(params.spin) * 20.0
    if params.spin > 0:
        fitness += params.spin * 18.0 * scenario.aggression
    else:
        fitness += abs(params.spin) * 10.0 * (1.0 - scenario.aggression)
    fitness += max(0.0, 2.4 - abs(params.arc_height - 2.2)) * 18.0
    fitness += min(opponent_distance, 8.0) * (18.0 + scenario.aggression * 12.0)
    fitness -= target_error * 42.0
    fitness -= abs(params.launch_angle_deg - (14.0 + scenario.aggression * 7.0)) * 3.2

    if cleared_net:
        fitness += 160.0
        if net_y < net_required_height + 0.35:
            fitness -= (net_required_height + 0.35 - net_y) * 75.0
        elif net_y > net_required_height + 2.2:
            fitness -= (net_y - net_required_height - 2.2) * 35.0
    else:
        fitness -= 260.0 + (net_required_height - net_y) * 80.0

    if in_court:
        fitness += 180.0
    else:
        fitness -= 360.0
        if abs(end_x) > half_width:
            fitness -= (abs(end_x) - half_width) * 95.0
        if end_z > 0:
            fitness -= end_z * 95.0
        if end_z < -half_length:
            fitness -= (-half_length - end_z) * 95.0

    return ShotResult(
        scenario=scenario,
        parameters=params,
        trajectory=trajectory,
        net_height_at_crossing=net_y,
        cleared_net=cleared_net,
        in_court=in_court,
        target_error=target_error,
        opponent_distance=opponent_distance,
        fitness=fitness,
    )


def standard_scenarios(config: CourtConfig = COURT_CONFIG) -> list[ShotScenario]:
    """Training situations across target and opponent placements."""
    half_width = config.singles_width / 2
    return [
        ShotScenario(1.9, 8.9, 1.15, 31.0, 0.45, 1.8, -5.3, -half_width + 0.85, -8.2, 0.9),
        ShotScenario(-1.4, 8.4, 1.35, 27.0, -0.35, -1.6, -5.0, half_width - 0.9, -8.0, 0.85),
        ShotScenario(0.5, 7.8, 1.8, 22.0, 0.2, 2.2, -4.8, -2.9, -6.8, 0.65),
        ShotScenario(-0.8, 8.2, 1.05, 25.0, -0.55, 0.4, -6.0, 2.7, -7.6, 0.7),
        ShotScenario(1.1, 8.7, 1.25, 33.0, 0.65, -0.8, -5.4, -3.25, -7.65, 0.95),
    ]
