# STATUS.md

## Current Status

Final project direction is locked:

**WinnerMind Tennis Shot Strategy AI**

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
  - `models/winnermind_best.weights.h5`
  - `reports/training_history.csv`
  - `reports/fitness.png`
  - `reports/best_shot.json`
  - `reports/best_shot.js`
- Built final visualization:
  - `reports/tennis_court_realistic_3d.html`
- The 3D visualization reads `reports/best_shot.js` when opened with `file:///...`, and `reports/best_shot.json` when served from a local server.
- Added `python -m neuropilot.demo`:
  - creates a random incoming ball situation
  - loads the trained WinnerMind weights
  - lets the neural network choose the return shot
  - scores the shot with the fitness function
  - updates the 3D visualization data
- Added random-shot support to the optional Gradio GUI.
- Added a sample `reports/best_shot.json` so the visualization works immediately; training overwrites it with real model output.
- Removed old experimental preview files to keep the project focused.
- Verified Python source syntax with `python -m compileall src`.
- Verified the final 3D HTML page can be opened directly at:
  - `file:///C:/Users/ASUS-H170M/Desktop/from_idea/exercsie3/reports/tennis_court_realistic_3d.html`
- Optional local server path, only if needed:
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
- incoming ball height
- incoming ball speed
- incoming ball spin
- opponent x position
- opponent z position
- aggression level

The neural network outputs:

- power
- launch angle
- arc height
- signed spin:
  - positive = topspin
  - negative = slice/backspin
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

Run the random incoming ball demo:

```powershell
python -m neuropilot.demo --seed 12
```

Then inspect the generated training outputs:

- `reports/fitness.png`
- `reports/training_history.csv`
- `reports/best_shot.json`
- `models/winnermind_best.weights.h5`

Then open the 3D visualization:

```text
file:///C:/Users/ASUS-H170M/Desktop/from_idea/exercsie3/reports/tennis_court_realistic_3d.html
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
  - Keras model input shape: `(None, 8)`
  - Keras model output shape: `(None, 6)`
  - One simulated shot produced valid trajectory and fitness data.
  - Tiny 2-generation genetic training run completed.
  - Tiny run verified weights, history CSV, and best-shot JSON export.
- Added realistic court bounds and spin behavior:
  - run-off margins in config
  - net center/post heights
  - signed spin where positive means topspin and negative means slice/backspin
  - 3D ball rotation now changes direction based on spin
  - 3D outer court dimensions now match recommended run-off footprint more closely
- Added incoming ball situation:
  - model now receives incoming height, speed, and spin
  - best-shot JSON exports incoming ball values
  - 3D animation shows the incoming ball first and then the AI return shot
- Fixed the 3D animation timing so the incoming ball bounces once on our side and is returned immediately, instead of bouncing twice before the AI shot.
- Updated training so `reports/fitness.png`, `reports/training_history.csv`, and `reports/best_shot.json` refresh after every generation, matching the assignment request for a chart updated during training.
- Full training completed through generation 40.
- Training generated:
  - `models/winnermind_best.weights.h5`
  - `reports/training_history.csv`
  - `reports/fitness.png`
  - `reports/best_shot.json`
- Best generation result reached best fitness around `722.87`.
- Latest `best_shot.json` contains a valid in-court shot that clears the net.
- Added `reports/best_shot.js` so the trained WinnerMind shot loads correctly when the HTML is opened directly from the file system.
- Added a random incoming ball demo so each run can show a different tennis situation and a model-controlled return shot.

## Suggested Next Commit

```text
feat: add random incoming ball demo
```

## Proposed Next Feature Commit

```text
feat: add opponent handedness to shot strategy
```
