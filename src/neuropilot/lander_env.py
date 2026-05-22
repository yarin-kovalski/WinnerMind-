"""2D lander simulation environment.

This file will contain the physics world used to evaluate each neural network.
The first implementation milestone is a deterministic simulation that can run
with a random controller before the genetic algorithm is connected.
"""

from __future__ import annotations

from dataclasses import dataclass

from neuropilot.config import ENV_CONFIG, EnvironmentConfig


@dataclass
class LanderState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    fuel: float
    steps: int = 0


@dataclass
class StepResult:
    state: LanderState
    done: bool
    landed: bool
    crashed: bool


class LanderEnv:
    """Small deterministic 2D landing simulator."""

    def __init__(self, config: EnvironmentConfig = ENV_CONFIG) -> None:
        self.config = config
        self.state = self.reset()

    def reset(self) -> LanderState:
        self.state = LanderState(
            x=self.config.width * 0.25,
            y=self.config.height * 0.85,
            vx=0.0,
            vy=0.0,
            angle=0.0,
            angular_velocity=0.0,
            fuel=self.config.initial_fuel,
        )
        return self.state

    def get_observation(self) -> list[float]:
        """Return normalized model inputs for the current state."""
        state = self.state
        return [
            (state.x - self.config.landing_pad_x) / self.config.width,
            state.y / self.config.height,
            state.vx / 10.0,
            state.vy / 10.0,
            state.angle / 3.14159,
            state.angular_velocity / 5.0,
            state.fuel / self.config.initial_fuel,
            abs(state.x - self.config.landing_pad_x) / self.config.width,
        ]

    def step(self, main_thrust: float, steering: float) -> StepResult:
        """Advance the simulation by one step.

        `main_thrust` and `steering` are expected to be in the range [-1, 1].
        Full physics and collision details will be expanded in the next step.
        """
        state = self.state
        fuel_available = state.fuel > 0
        thrust = max(0.0, min(1.0, main_thrust)) if fuel_available else 0.0
        steer = max(-1.0, min(1.0, steering)) if fuel_available else 0.0

        fuel_used = thrust * 0.12 + abs(steer) * 0.03
        next_fuel = max(0.0, state.fuel - fuel_used)
        next_angle = state.angle + state.angular_velocity
        next_angular_velocity = state.angular_velocity + steer * 0.015
        next_vx = state.vx + steer * 0.02
        next_vy = state.vy + self.config.gravity + thrust * 0.16
        next_x = state.x + next_vx
        next_y = state.y + next_vy

        landed = (
            next_y <= 0
            and abs(next_x - self.config.landing_pad_x) < 5.0
            and abs(next_vx) < 1.0
            and abs(next_vy) < 1.2
            and abs(next_angle) < 0.25
        )
        crashed = next_y <= 0 and not landed
        out_of_bounds = next_x < 0 or next_x > self.config.width or next_y > self.config.height * 1.2
        done = landed or crashed or out_of_bounds or state.steps + 1 >= self.config.max_steps

        self.state = LanderState(
            x=next_x,
            y=next_y,
            vx=next_vx,
            vy=next_vy,
            angle=next_angle,
            angular_velocity=next_angular_velocity,
            fuel=next_fuel,
            steps=state.steps + 1,
        )
        return StepResult(state=self.state, done=done, landed=landed, crashed=crashed)
