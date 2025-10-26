#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import os
import time
import argparse
import csv

# ------------------------------------------------------------------------
# 1) Always create a 1200-step speed dataset
# ------------------------------------------------------------------------
DATA_LEN = 1200
CSV_FILE = "speed_profile.csv"

# Force-generate a 1200-step sinusoidal + noise speed profile
speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)
df_fake = pd.DataFrame({"speed": speeds})
df_fake.to_csv(CSV_FILE, index=False)
print(f"Created {CSV_FILE} with {DATA_LEN} steps.")

df = pd.read_csv(CSV_FILE)
full_speed_data = df["speed"].values
assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps after generation."

# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
# ------------------------------------------------------------------------
def chunk_into_episodes(data, chunk_size):
    """
    Splits `data` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    """
    episodes = []
    start = 0
    while start < len(data):
        end = start + chunk_size
        chunk = data[start:end]
        episodes.append(chunk)
        start = end
    return episodes

def select_algo(algo_name, train_env, device, learning_rate=3e-4, batch_size=256):

    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)

    if algo_name == "SAC":
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=device,

            # Model specific params
            learning_rate=learning_rate,
            batch_size=batch_size, # default is 256
            buffer_size=200000,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
        )
    elif algo_name == "PPO":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=device,

            # Model specific params
            learning_rate=learning_rate,
            batch_size=64, # default is 64
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
        )
    elif algo_name == "TD3":
        model = TD3(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=device,

            # Model specific params
            learning_rate=learning_rate, # default is 1e-3
            batch_size=256,
            buffer_size=1000000,
            tau=0.005,
            gamma=0.99,
            # policy_delay=2,
            train_freq=1,
        )
    elif algo_name == "DDPG":
        model = DDPG(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=device,

            # Model specific params
            learning_rate=learning_rate,
            batch_size=256,
            buffer_size=1000000,
            tau=0.005,
            gamma=0.99,
        )
    else:
        raise ValueError("Invalid algorithm selected!")

    return model

def compute_metrics(algo_name, chunk_size, learning_rate, batch_size, reference_speeds, predicted_speeds, rewards, total_timesteps, output_csv):
    """
    Compute performance metrics after testing.
    """
    ref = np.array(reference_speeds)
    pred = np.array(predicted_speeds)

    mae = mean_absolute_error(ref, pred)
    mse = mean_squared_error(ref, pred)
    rmse = np.sqrt(mse)

    # Convergence rate (very rough): average reward improvement per 1k steps
    # You can refine this later if you log intermediate rewards.
    convergence_rate = np.mean(rewards[-100:]) - np.mean(rewards[:100])
    convergence_rate /= total_timesteps / 1000  # normalize per 1k steps

    print(f"[METRICS] MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, ConvergenceRate={convergence_rate:.6f}")

    # Append results to CSV
    fieldnames = ["Algorithm", "ChunkSize", "LearningRate", "BatchSize", "MAE", "MSE", "RMSE", "ConvergenceRate"]
    new_row = {
        "Algorithm": algo_name,
        "ChunkSize": chunk_size,
        "LearningRate": learning_rate,
        "BatchSize": batch_size,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "ConvergenceRate": convergence_rate
    }

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(new_row)

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):
    """
    Speed-following training environment:
      - The dataset is split into episodes of length `chunk_size`.
      - Each reset(), we pick one chunk at random.
      - action: acceleration in [-3,3]
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, episodes_list, delta_t=1.0):
        super().__init__()
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t

        # Actions, Observations
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2,), dtype=np.float32)

        # Episode-specific
        self.current_episode = None
        self.episode_len = 0
        self.step_idx = 0
        self.current_speed = 0.0
        self.ref_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick random chunk from episodes_list
        ep_idx = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Initialize
        self.current_speed = 0.0
        self.ref_speed = self.current_episode[self.step_idx]

        obs = np.array([self.current_speed, self.ref_speed], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        self.ref_speed = self.current_episode[self.step_idx]
        error = abs(self.current_speed - self.ref_speed)
        reward = -error

        self.step_idx += 1
        terminated = (self.step_idx >= self.episode_len)
        truncated = False

        obs = np.array([self.current_speed, self.ref_speed], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Speed-following testing environment:
      - We run through the entire 1200-step dataset in one go.
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, full_data, delta_t=1.0):
        super().__init__()
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2,), dtype=np.float32)

        self.idx = 0
        self.current_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.current_speed = 0.0
        ref_speed = self.full_data[self.idx]
        obs = np.array([self.current_speed, ref_speed], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        ref_speed = self.full_data[self.idx]
        error = abs(self.current_speed - ref_speed)
        reward = -error

        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([self.current_speed, ref_speed], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 4) CustomLoggingCallback (optional)
# ------------------------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, log_name="training_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_name)
        self.episode_rewards = []
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward'])

    def _on_step(self):
        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        self.episode_rewards.append(reward)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward])
            self.logger.record("reward/average_reward", avg_reward)
            self.episode_rewards.clear()

        return True

class ModelEvaluationEnv():

    def __init__(self, algo_name, learning_rate, batch_size, total_timesteps, chunk_size, log_dir, logger, episodes_list):
        self.algo_name = algo_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.logger = logger
        self.episodes_list = episodes_list

        self.model = None

    # ------------------------------------------------------------------------
    # Train a model
    # ------------------------------------------------------------------------
    def train(self):
        # Name of model trained with the unique set of parameters
        trained_model_name = f"{self.algo_name}_lr={self.learning_rate}_bs={self.batch_size}_cs={self.chunk_size}"

        # Check if pre-trained model exists for the given set of parameters
        # If it doesn't exist, start training
        trained_model_path = f"{self.log_dir}/{trained_model_name}"
        if os.path.exists(f"{self.log_dir}/{trained_model_name}"):
            print(f"[INFO] {trained_model_name} already exists! Loading pretrained model")
            if algo_name == "SAC":
                self.model = SAC.load(trained_model_path)
            elif algo_name == "PPO":
                self.model = PPO.load(trained_model_path)
            elif algo_name == "TD3":
                self.model = TD3.load(trained_model_path)
            elif algo_name == "DDPG":
                self.model = DDPG.load(trained_model_path)
            else:
                raise ValueError("Invalid algorithm selected!")
        else:
            print(f"[INFO] {trained_model_name} doesn't exist. Starting training")
            self.model = self.train_model()

    def train_model(self):

        # 5B) Create the TRAIN environment
        def make_train_env():
            return TrainEnv(self.episodes_list, delta_t=1.0)

        train_env = DummyVecEnv([make_train_env])

        # 5C) Build the model (SAC with MlpPolicy)
        device = torch.device("cuda" if torch.cuda.is_available() and self.algo_name != "PPO" else "cpu")
        print(f"Training on device: {device}")

        # Select the algorithm and associated hyperparameters
        self.model = select_algo(self.algo_name, train_env, device, self.learning_rate, self.batch_size)
        self.model.set_logger(self.logger)

        callback = CustomLoggingCallback(self.log_dir)

        print(f"[INFO] Start training for {self.total_timesteps} timesteps...")
        start_time = time.time()
        self.model.learn(
            total_timesteps=self.total_timesteps,
            log_interval=100,
            callback=callback
        )
        end_time = time.time()
        print(f"[INFO] Training finished in {end_time - start_time:.2f}s")

        # 5D) Save the model
        save_path = os.path.join(self.log_dir, trained_model_name)
        self.model.save(save_path)
        print(f"[INFO] Model saved to: {save_path}.zip")

# ------------------------------------------------------------------------
# Declare project tasks based on assignment document
# ------------------------------------------------------------------------
def task1():
    print("running test 1")
    pass
    
def run_from_command_line(algo_name, batch_size, chunk_size, learning_rate, total_timesteps, log_dir, logger):

    episodes_list = chunk_into_episodes(full_speed_data, chunk_size)

    print(f"[INFO] Using algorithm: {algo_name}")
    print(f"[INFO] Using chunk_size = {chunk_size}")
    print(f"[INFO] Using learning_rate = {learning_rate}")
    print(f"Number of episodes: {len(episodes_list)} (some leftover if 1200 not divisible by {chunk_size})")
    print(f"[INFO] Using batch_size = {batch_size}")

    model_env = ModelEvaluationEnv(algo_name=algo_name, 
                                   batch_size=batch_size,
                                   learning_rate=learning_rate,
                                   chunk_size=chunk_size,
                                   total_timesteps=total_timesteps,
                                   log_dir=log_dir,
                                   logger=logger,
                                   episodes_list=episodes_list
                                   )

    # Train model based on given parameters
    model_env.train()
    model = model_env.model

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, delta_t=1.0)

    obs, _ = test_env.reset()
    predicted_speeds = []
    reference_speeds = []
    rewards = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        predicted_speeds.append(obs[0])  # current_speed
        reference_speeds.append(obs[1])  # reference_speed
        rewards.append(reward)
        if terminated or truncated:
            break

    avg_test_reward = np.mean(rewards)
    print(f"[TEST] Average reward over 1200-step test: {avg_test_reward:.3f}")
    metrics_csv = os.path.join(log_dir, "metrics_summary.csv")
    compute_metrics(algo_name, chunk_size, learning_rate, batch_size, reference_speeds, predicted_speeds, rewards, total_timesteps, metrics_csv)


    # Plot the entire test
    plt.figure(figsize=(10, 5))
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Test on full 1200-step dataset (chunk_size={chunk_size})")
    plt.legend()
    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{algo_name}_chunksize={chunk_size}_lr={str(learning_rate)}_bs={batch_size}.png")

# ------------------------------------------------------------------------
# 5) Main: user sets chunk_size from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_chunk_training",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Episode length for training (e.g. 50, 100, 200)."
    )
    parser.add_argument(
        "-a, --algorithm",
        dest="algorithm",
        type=str,
        default="SAC",
        choices=["SAC", "PPO", "TD3", "DDPG"],
        help="Select the RL algorithm to use."
    )
    parser.add_argument(
        "-lr, --learning-rate",
        dest="learning_rate",
        type=float,
        default=3e-4,
        help="Select the learning rate for training."
    )
    parser.add_argument(
        "-bs, --batch-size",
        dest="batch_size",
        type=int,
        default=256,
        help="Select the batch size for training."
    )
    parser.add_argument(
        "-ts, --timesteps",
        dest="timesteps",
        type=int,
        default=100_000,
        help="Select the total number of time steps."
    )
    parser.add_argument(
        "--test",
        dest="test",
        type=str,
        default="cli",
        help="Select the test to run."
    )
    args = parser.parse_args()

    # Create logger
    log_dir = args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    # Parse args
    algo_name = args.algorithm
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    
    learning_rate = args.learning_rate
    total_timesteps = args.timesteps
    test = args.test

    # Select test to run
    if (test == "cli"):
        run_from_command_line(algo_name, batch_size, chunk_size, learning_rate, total_timesteps, log_dir, logger)
    elif test == "test1":
        test1()
    else:
        raise ValueError("Invalid test selected!")

if __name__ == "__main__":
    main()
