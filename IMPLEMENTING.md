# IMPLEMENTING.md

## Implementation Plan For Exercise 3

Goal: get a high grade by building a complete, original, demonstrable Keras neural-network project using a genetic algorithm.

## Chosen Concept

**NeuroPilot**: evolve a neural network that controls a 2D spacecraft landing simulator.

The neural network receives the lander's current state and outputs control decisions. The genetic algorithm improves the model weights over many generations.

## Neural Network Design

Input features:

- horizontal position relative to landing pad
- vertical position
- horizontal velocity
- vertical velocity
- angle
- angular velocity
- remaining fuel
- distance to landing pad

Outputs:

- main thrust strength
- left/right steering or rotation control

Initial architecture:

- Keras `Sequential` or functional model
- Dense layer, 32 units, ReLU
- Dense layer, 32 units, ReLU
- Dense output layer, 2 units, `tanh` or `sigmoid` depending on action encoding

## Genetic Algorithm

Each individual is one Keras model's weights.

Training loop:

1. Create a population of random neural networks.
2. Run each network in the lander simulation.
3. Score each network with a fitness function.
4. Keep the top performers.
5. Create children by combining parent weights.
6. Mutate some child weights.
7. Repeat for many generations.
8. Save the best model weights.

## Fitness Function

Reward:

- getting close to the landing pad
- slow vertical speed near the ground
- slow horizontal speed near the pad
- upright angle
- successful landing
- fuel remaining
- surviving longer without leaving bounds

Penalty:

- crashing
- high-speed impact
- drifting too far from the target
- rotating too much
- wasting fuel
- leaving the simulation bounds

The first version should be simple and stable. We can improve it after seeing training behavior.

## Visual Progress

Required by assignment:

- show how fitness improves over generations

Implementation:

- save fitness history to `reports/training_history.csv`
- generate a chart with best fitness and average fitness
- update the chart during training or at least after every generation
- optionally show a live matplotlib window

## GUI / Wow Factor

Recommended final GUI:

- Gradio app with buttons:
  - Train
  - Run Best Model
  - Load Saved Model
- Show:
  - current generation
  - best fitness
  - fitness chart
  - animated or step-by-step lander path

If Gradio animation is too slow, use matplotlib images/GIFs or a local visual demo script.

## Baby-Step Milestones

1. Create project structure and planning files.
2. Implement lander physics environment without AI.
3. Add a random controller demo to verify physics.
4. Build the Keras model factory.
5. Implement fitness scoring for one model.
6. Implement genetic algorithm selection, crossover, and mutation.
7. Train for a small number of generations and save history.
8. Save and load best model weights.
9. Add plotting for training progress.
10. Add visual demo of the trained lander.
11. Add Gradio GUI or polished CLI demo.
12. Test from a clean run and prepare video instructions.

## Quality Checklist For 100

- The project clearly uses Keras.
- The training method is clearly a genetic algorithm.
- The fitness chart is visible and understandable.
- The model weights are saved.
- The demo loads the saved model and proves training worked.
- The idea feels original compared to classic cat/dog/image examples.
- The code is organized and readable.
- The video shows the assignment requirements one by one.
- The README explains setup, training, demo, and submission.
- `STATUS.md` documents progress honestly.

## Suggested Git Commit Plan

- `chore: add exercise 3 project plan and structure`
- `feat: implement lander simulation environment`
- `feat: add keras neural pilot model`
- `feat: evaluate neural pilot fitness`
- `feat: train neural pilot with genetic algorithm`
- `feat: save model weights and training chart`
- `feat: add trained model demo`
- `feat: add gui for training and demo`
- `docs: add run instructions and video checklist`
