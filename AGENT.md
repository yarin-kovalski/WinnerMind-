# AGENT.md

## Program Vision

This project implements Exercise 3 from Assignment 3: a Python program that trains a neural network using Keras.

Our chosen idea is **NeuroPilot: a genetic-algorithm-trained neural network that learns to land a small 2D spacecraft safely**.

The program will evolve many randomly initialized Keras neural networks. Each network controls a simulated lander by choosing actions such as thrust and steering. A fitness function rewards landers that:

- land near the target pad
- reduce vertical and horizontal speed before touchdown
- stay upright
- use fuel efficiently
- avoid crashing or flying away

This follows the assignment's second training option: **Genetic Algorithm**.

## Why This Idea Fits The Assignment

- Uses a Keras neural network.
- Trains without a labeled dataset by evolving weights over generations.
- Has a clear fitness function.
- Shows a visual chart of fitness improvement during training.
- Saves the trained model weights to a file after training.
- Can be demonstrated as a real interactive program with a "wow" factor.
- Has natural video material for the assignment competition: watch the lander improve over generations and then let the trained pilot fly.

## Target User Experience

The final project should feel like a small AI lab:

- Run training and watch a live fitness chart.
- See the best lander from each generation in a visual simulation.
- Save the best model weights.
- Load the trained model and run a demo flight.
- Optionally open a Gradio GUI that lets a user train, test, and watch the neural pilot.

## Core Requirements From The Assignment

- Create a Python program that trains a neural network using Keras.
- Choose an interesting original problem.
- Use either supervised learning or a genetic algorithm.
- If using a genetic algorithm:
  - start with randomly weighted neural networks
  - score each network using a fitness function
  - select the best networks
  - breed and mutate them over multiple generations
  - show how fitness improves during training
- Save the model weights into a file when training finishes.
- Create a real program with value or a strong "wow" factor.
- GUI is allowed and recommended; Gradio is a good option.
- Record a high quality video showing Exercise 3 running properly.

## Proposed Project Structure

```text
.
├── AGENT.md
├── IMPLEMENTING.md
├── STATUS.md
├── requirements.txt
├── .gitignore
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   └── neuropilot/
│       ├── __init__.py
│       ├── config.py
│       ├── lander_env.py
│       ├── model.py
│       ├── genetic_algorithm.py
│       ├── train.py
│       ├── demo.py
│       ├── gui.py
│       └── plotting.py
└── tests/
    └── .gitkeep
```

## Development Rules

- Work in baby steps, as requested by the assignment.
- Update `STATUS.md` after every meaningful progress step.
- Keep commits meaningful once a Git repository is initialized.
- Review diffs before each commit.
- Do not commit virtual environments, cache folders, generated videos, or very large model artifacts unless needed.
- Keep the project runnable from a clean checkout using `requirements.txt`.

## Definition Of Done

The project is done when:

- training runs successfully for multiple generations
- the chart shows best and average fitness improving
- trained Keras weights are saved under `models/`
- a demo can load the saved weights and run the lander
- a GUI or clear visual demo exists
- instructions explain how to run everything
- the final video can show the full story clearly
