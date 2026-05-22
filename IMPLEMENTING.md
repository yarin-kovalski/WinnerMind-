# IMPLEMENTING.md

## Final Project

**AceMind Tennis Shot Strategy AI**

We train a Keras neural network using a genetic algorithm. The neural network learns to choose tennis shot parameters for different tactical situations.

## Neural Network

Input features:

- incoming ball x position
- incoming ball z position
- opponent x position
- opponent z position
- aggression level

Outputs:

- power
- launch angle
- topspin
- target x position
- target z position

## Training Method

We use the assignment's option 2: **Genetic Algorithm**.

Training loop:

1. Create many random Keras neural networks.
2. Each network predicts shot parameters.
3. The simulator calculates the ball trajectory.
4. The fitness function scores the shot.
5. Keep the best networks.
6. Mix parent weights with crossover.
7. Mutate child weights.
8. Repeat for multiple generations.
9. Save the best model weights.
10. Export the best shot to `reports/best_shot.json`.

## Fitness Function

A shot is good if:

- it clears the net
- it lands inside the opponent court
- it lands near the desired target
- it lands away from the opponent
- it has useful speed and spin

A shot is bad if:

- it does not clear the net
- it lands out
- it is far from the target
- it lands near the opponent
- it uses unrealistic launch/power

## Generated Files

Training outputs:

- `models/acemind_best.weights.h5`
- `reports/training_history.csv`
- `reports/fitness.png`
- `reports/best_shot.json`

Visualization:

- `reports/tennis_court_realistic_3d.html`

## Final Video Plan

Show:

- Exercise 3 requirement: Keras neural network
- option 2: genetic algorithm
- explain no dataset is needed because simulation creates fitness scores
- run training
- show generation improvement
- show `fitness.png`
- show saved model weights
- show `best_shot.json`
- open the 3D tennis animation
- explain that the ball movement uses the trained model parameters

## Two-Hour Finish Plan

1. Install dependencies.
2. Run training.
3. Inspect `best_shot.json`.
4. Open 3D animation.
5. If motion looks bad, tune only the fitness/scaling values.
6. Record video.
