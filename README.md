# Reinforcement Learning ACC Assignment

This goal of this assignment is to implement adaptive cruise control using reinforcement learning and evaluate the results with different hyperparameters.
This script can be run by using the commands in the following sections

## Run All Hyperparameter Sweeps

```
python3 ./rl.py --task all
```

## Batch Size Sweep

```
python3 ./rl.py --task batch_size_test
```

## Learning Rate Sweep

```
python3 ./rl.py --task lr_test
```

## Entropy Coefficient Sweep

```
python3 ./rl.py --task ent_coef_test
```

## Episode Length Sweep

```
python3 ./rl.py --task episode_test
```

## Run All Models with Best Hyperparameters

```
python3 ./rl.py --task best_params_test
```
