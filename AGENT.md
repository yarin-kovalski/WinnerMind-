# AGENT.md

## Program Vision

This project implements Exercise 3 from Assignment 3: a Python program that trains a neural network using Keras.

Final idea: **AceMind Tennis Shot Strategy AI**.

AceMind trains a Keras neural network with a genetic algorithm. The model receives a tennis game situation and outputs shot parameters:

- shot power
- launch angle
- topspin
- target x position
- target z position

A tennis simulation scores each shot. The best neural networks survive, breed, mutate, and improve over generations.

## Why This Fits The Assignment

- Uses Python.
- Uses a Keras neural network.
- Uses option 2: genetic algorithm.
- Does not require a dataset because training happens through simulation and fitness scoring.
- Shows training progress with a fitness chart.
- Saves trained model weights.
- Exports the best shot parameters to `reports/best_shot.json`.
- Shows the trained result in a realistic 3D tennis court animation.

## How The Algorithm Knows A Shot Is Good

The model is scored with a fitness function.

Reward:

- ball crosses above the net
- ball lands inside the opponent side of the court
- ball lands close to the desired target
- ball lands far from the opponent
- shot has useful power and spin

Penalty:

- ball hits or fails to clear the net
- ball lands out
- ball lands far from target
- shot is too weak or unrealistic
- target is easy for the opponent to reach

## Final User Experience

1. Run training.
2. Watch generation fitness improve.
3. Save the best model weights.
4. Save `reports/best_shot.json`.
5. Open the 3D tennis court animation.
6. The animation reads the trained parameters and shows the ball moving according to them.

## Development Rules

- Work in baby steps.
- Update `STATUS.md` after meaningful progress.
- Keep commits meaningful once Git is initialized.
- Do not commit `.venv`, caches, generated videos, or other large unneeded files.
- Keep the project runnable from `requirements.txt`.

## Definition Of Done

- `python -m neuropilot.train` trains multiple generations.
- `models/acemind_best.weights.h5` is saved.
- `reports/training_history.csv` is saved.
- `reports/fitness.png` is saved.
- `reports/best_shot.json` is saved.
- `reports/tennis_court_realistic_3d.html` displays the trained shot parameters.
- Final video shows training, chart, saved files, and 3D animation.
