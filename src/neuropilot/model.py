"""Keras model factory for AceMind Tennis Shot Strategy AI."""

from __future__ import annotations

from keras import Sequential
from keras.layers import Dense, Input


def build_pilot_model() -> Sequential:
    """Map a game situation to shot parameters.

    Inputs:
    - incoming ball x
    - incoming ball z
    - opponent x
    - opponent z
    - strategy aggression

    Outputs:
    - power
    - launch angle
    - arc height
    - topspin
    - target x
    - target z
    """
    return Sequential(
        [
            Input(shape=(5,)),
            Dense(32, activation="relu"),
            Dense(32, activation="relu"),
            Dense(6, activation="sigmoid"),
        ]
    )
