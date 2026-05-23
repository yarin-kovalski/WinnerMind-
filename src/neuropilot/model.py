"""Keras model factory for AceMind Tennis Shot Strategy AI."""

from __future__ import annotations

from keras import Sequential
from keras.layers import Dense, Input


def build_pilot_model() -> Sequential:
    """Map a game situation to shot parameters.

    Inputs:
    - incoming ball x
    - incoming ball z
    - incoming ball height
    - incoming ball speed
    - incoming ball spin
    - opponent x
    - opponent z
    - strategy aggression

    Outputs:
    - power
    - launch angle
    - arc height
    - signed spin: negative means slice/backspin, positive means topspin
    - target x
    - target z
    """
    return Sequential(
        [
            Input(shape=(8,)),
            Dense(32, activation="relu"),
            Dense(32, activation="relu"),
            Dense(6, activation="sigmoid"),
        ]
    )
