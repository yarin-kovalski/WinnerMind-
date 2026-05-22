# AceMind Tennis Shot Strategy AI

Exercise 3 implementation for Assignment 3: training a Keras neural network with a genetic algorithm.

## Idea

AceMind learns tennis shot strategy in simulation. The model receives a game situation and outputs:

- power
- launch angle
- topspin
- target x position
- target z position

The genetic algorithm scores each model using a fitness function and evolves better models over generations.

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

- `models/acemind_best.weights.h5`
- `reports/training_history.csv`
- `reports/fitness.png`
- `reports/best_shot.json`

## Show Best Shot Data

```powershell
python -m neuropilot.demo
```

## 3D Visualization

Start a local server from the project root:

```powershell
python -m http.server 8000
```

Open:

```text
http://localhost:8000/reports/tennis_court_realistic_3d.html
```

The animation reads `reports/best_shot.json` when it exists. If you open it before training, it uses preview values.

## Video Checklist

- Explain: Keras neural network + genetic algorithm.
- Explain: no dataset is needed because shots are scored in simulation.
- Run training.
- Show generations improving.
- Show `reports/fitness.png`.
- Show `models/acemind_best.weights.h5`.
- Show `reports/best_shot.json`.
- Show the 3D court animation using the trained shot parameters.
