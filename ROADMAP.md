# ROADMAP.md

## Where We Are

We have chosen the final project:

**WinnerMind Tennis Shot Strategy AI**

This is Assignment 3, Exercise 3, option 2: **Genetic Algorithm**.

Current status:

- Project structure exists.
- Keras model code exists.
- Tennis shot simulation exists.
- Genetic algorithm trainer exists.
- Fitness chart output exists.
- Best shot JSON export exists.
- Realistic 3D court visualization exists.
- The 3D visualization reads `reports/best_shot.json`.
- Runtime dependencies are not installed yet in the current Python environment.
- Real training has not been run yet on this machine.

## What The Exercise Requires

We need to show:

- Python program
- Keras neural network
- genetic algorithm training
- fitness function
- multiple generations
- visual chart of training progress
- saved model weights
- real/wow program
- final video showing the program working

## How Our Project Answers It

The neural network receives:

- incoming ball x position
- incoming ball z position
- opponent x position
- opponent z position
- aggression level

The neural network outputs:

- power
- launch angle
- arc height
- topspin
- target x
- target z

The simulator calculates the ball path and scores it.

The genetic algorithm improves the neural networks over generations.

The best model is saved and the best shot is shown in 3D.

## Step-by-Step Plan

### Step 1 - Finalize project direction

Status: done

Commit name:

```text
docs: finalize winnermind shot strategy plan
```

### Step 2 - Add explicit shot height feature

Status: done

Why:

The net is important. The AI should learn how high the ball should travel, not only where it should land.

Commit name:

```text
feat: add arc height to shot strategy model
```

### Step 3 - Install dependencies

Status: done

Command:

```powershell
pip install -r requirements.txt
```

Commit name:

```text
chore: install training dependencies
```

Usually you do not commit installed packages or `.venv`.

Verified imports:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Gradio
- Pillow

### Optional Feature - Opponent handedness

Status: proposed

Idea:

Add whether the opponent is left-handed or right-handed. This can affect the opponent's weaker side and make the AI choose smarter targets.

Potential model input:

- opponent handedness: `0 = left-handed`, `1 = right-handed`

Potential fitness behavior:

- reward shots to the opponent's weaker side
- make the 3D panel show opponent handedness

Commit name:

```text
feat: add opponent handedness to shot strategy
```

### Step 3A - Add realistic court bounds and spin behavior

Status: done

What changed:

- Added official-style run-off margins to config for visualization.
- The red/orange margin is treated as out-of-play; only the court rectangle can score as in.
- Added center/post net heights.
- Added signed spin:
  - positive = topspin
  - negative = slice/backspin
- Fitness now accounts for spin style and net clearance more accurately.

Commit name:

```text
feat: add realistic court bounds and spin behavior
```

### Step 3B - Add incoming ball situation

Status: done

What changed:

- Added incoming ball height, speed, and spin to model inputs.
- Exported incoming ball values to `best_shot.json`.
- 3D animation now shows the incoming ball first, then the AI return shot.

Commit name:

```text
feat: animate incoming ball before ai return
```

### Step 4 - Verify training pipeline

Status: done

What was checked:

- Keras model input shape is `(None, 10)`.
- Keras model output shape is `(None, 6)`.
- One simulated shot produced fitness and trajectory data.
- A tiny 2-generation genetic training run completed.
- Tiny test produced weights, history CSV, and best-shot JSON.

Commit name:

```text
test: verify winnermind training pipeline
```

### Step 5 - Run full training

Status: not done

Command:

```powershell
$env:PYTHONPATH="src"
python -m neuropilot.train
```

Expected outputs:

- `models/winnermind_best.weights.h5`
- `reports/training_history.csv`
- `reports/fitness.png`
- `reports/best_shot.json`

Commit name:

```text
feat: train winnermind shot strategy model
```

### Step 6 - Check visualization

Status: not done

Command:

```powershell
python -m http.server 8000
```

Open:

```text
http://localhost:8000/reports/tennis_court_realistic_3d.html
```

Commit name:

```text
feat: connect trained shot to 3d court
```

### Step 7 - Tune if needed

Status: not done

If the trained shot looks bad, tune:

- fitness rewards
- target scenarios
- mutation strength
- generation count

Commit name:

```text
tune: improve shot fitness behavior
```

### Step 8 - Record final video

Status: not done

Show:

- training command
- generations improving
- `fitness.png`
- saved model weights
- `best_shot.json`
- 3D court animation

Commit name:

```text
docs: add final video checklist
```
