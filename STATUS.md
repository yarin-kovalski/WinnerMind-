# STATUS.md

## Current Status

Final project direction is locked:

**AceMind Tennis Shot Strategy AI**

This implements Assignment 3, Exercise 3 using **option 2: Genetic Algorithm**.

The program trains a Keras neural network to choose tennis shot parameters. The trained result is visualized on a realistic 3D tennis court.

## Latest Progress

- Read the assignment PDF and focused on Exercise 3.
- Chose option 2, genetic algorithm, because the instructor explicitly says it is more interesting.
- Finalized the idea: a tennis shot strategy AI.
- Implemented Keras model factory.
- Implemented tennis shot simulation.
- Implemented fitness scoring.
- Implemented genetic algorithm training:
  - random population
  - evaluation
  - elite selection
  - crossover
  - mutation
  - multiple generations
- Training exports:
  - `models/acemind_best.weights.h5`
  - `reports/training_history.csv`
  - `reports/fitness.png`
  - `reports/best_shot.json`
- Built final visualization:
  - `reports/tennis_court_realistic_3d.html`
- The 3D visualization reads `reports/best_shot.json` when available.
- Added a sample `reports/best_shot.json` so the visualization works immediately; training overwrites it with real model output.
- Removed old experimental preview files to keep the project focused.
- Verified Python source syntax with `python -m compileall src`.
- Verified the final 3D HTML page is served at:
  - `http://localhost:8000/reports/tennis_court_realistic_3d.html`

## Assignment Requirements Tracked

- [x] Python program
- [x] Keras neural network
- [x] Original/interesting problem
- [x] Genetic algorithm training
- [x] Fitness function
- [x] Multiple generations
- [x] Visual chart of training progress
- [x] Saved model weights file
- [x] Real program / wow factor
- [x] GUI or visual demo
- [x] Video-ready final result

## How The Project Works

The neural network receives:

- incoming ball x position
- incoming ball z position
- opponent x position
- opponent z position
- aggression level

The neural network outputs:

- power
- launch angle
- topspin
- target x
- target z

The simulator checks if the shot:

- clears the net
- lands inside the court
- lands near the desired target
- lands far from the opponent
- uses useful power and spin

That score is the fitness. The genetic algorithm improves the model over generations.

## Next Step

Install dependencies and run training:

```powershell
pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m neuropilot.train
```

Then open:

```text
http://localhost:8000/reports/tennis_court_realistic_3d.html
```

## Progress Log

### 2026-05-22

- Created project structure.
- Planned the Exercise 3 solution.
- Iterated through visual ideas.
- Finalized the realistic 3D tennis court direction.
- Refactored Python code to match the final shot-strategy idea.
- Connected training output to the 3D visualization through `best_shot.json`.
- Added `ROADMAP.md` with step-by-step implementation plan and suggested commit names.
- Added explicit `arc_height` / `arcHeight` feature to the neural network output, simulator, fitness scoring, best-shot JSON, and 3D visualization.
- Created `.venv`.
- Installed dependencies from `requirements.txt`.
- Verified imports for TensorFlow, Keras, NumPy, Matplotlib, Gradio, and Pillow.
- Verified Python source syntax with the venv Python.
- Added opponent handedness as a proposed next feature in `ROADMAP.md`.
- Step 3 verification completed:
  - Keras model input shape: `(None, 5)`
  - Keras model output shape: `(None, 6)`
  - One simulated shot produced valid trajectory and fitness data.
  - Tiny 2-generation genetic training run completed.
  - Tiny run verified weights, history CSV, and best-shot JSON export.

## Suggested Next Commit

```text
test: verify acemind training pipeline
```

## Proposed Next Feature Commit

```text
feat: add opponent handedness to shot strategy
```
