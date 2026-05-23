# WinnerMind Tennis Shot Strategy AI

Exercise 3 implementation for Assignment 3: training a Keras neural network with a genetic algorithm.

## Idea

WinnerMind learns tennis shot strategy in simulation. The model receives a game situation:

- incoming ball position
- incoming ball height, speed, and spin
- opponent position
- opponent running direction
- aggression level

The model outputs:

- power
- launch angle
- arc height
- signed spin, where positive is topspin and negative is slice/backspin
- target x position
- target z position

The genetic algorithm scores each model using a fitness function and evolves better models over generations. Strong shots are rewarded when they clear the net, land in court, stay away from the opponent, and hit against the opponent's movement direction.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:PYTHONPATH="src"
```

## Train

```powershell
python -m neuropilot.train
```

Outputs:

- `models/winnermind_best.weights.h5`
- `reports/training_history.csv`
- `reports/fitness.png`
- `reports/best_shot.json`
- `reports/best_shot.js`

## Random Incoming Ball Demo

```powershell
python -m neuropilot.demo
```

This loads `models/winnermind_best.weights.h5`, creates a new random incoming ball, lets the trained neural network choose power, launch angle, arc height, spin, and target, scores the result with the fitness function, and refreshes `reports/best_shot.json` plus `reports/best_shot.js` for the 3D animation. The GUI button creates a fresh random ball every click.

## Optional GUI

```powershell
python -m neuropilot.gui
```

## 3D Visualization

Open directly:

```text
file:///C:/Users/ASUS-H170M/Desktop/from_idea/exercsie3/reports/tennis_court_realistic_3d.html
```

Or start a local server from the project root:

```powershell
python -m http.server 8000
```

Open:

```text
http://localhost:8000/reports/tennis_court_realistic_3d.html
```

The animation reads `reports/best_shot.js` when opened directly from `file:///...`, and falls back to `reports/best_shot.json` when served from a local server. If you open it before training, it uses preview values.

## Video Checklist

- Explain: Keras neural network + genetic algorithm.
- Explain: no dataset is needed because shots are scored in simulation.
- Run training.
- Show generations improving.
- Show `reports/fitness.png`.
- Show `models/winnermind_best.weights.h5`.
- Run `python -m neuropilot.demo` to show a new random incoming ball and the trained model response.
- Show `reports/best_shot.json`.
- Show the 3D court animation using the trained shot parameters.
