"""Keras model factory for the neural lander pilot."""

from __future__ import annotations

from keras import Sequential
from keras.layers import Dense, Input


def build_pilot_model() -> Sequential:
    """Create a small neural network that maps lander state to controls."""
    model = Sequential(
        [
            Input(shape=(8,)),
            Dense(32, activation="relu"),
            Dense(32, activation="relu"),
            Dense(2, activation="tanh"),
        ]
    )
    return model
