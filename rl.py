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

    def __init__(self, log_dir="logs_chunk_training", algo_name="SAC", learning_rate=3e-4, batch_size=256, total_timesteps=100_000, chunk_size=100, models_dir="trained_models"):
        self.algo_name = algo_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.models_dir = models_dir
        self.logger = configure(self.log_dir, ["stdout", "tensorboard"])
        self.trained_model_name = None

        self.episodes_list = chunk_into_episodes(full_speed_data, self.chunk_size)

        self.model = None

    def compute_metrics(self, reference_speeds, predicted_speeds, rewards):
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
        convergence_rate /= self.total_timesteps / 1000  # normalize per 1k steps

        print(f"[METRICS] MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, ConvergenceRate={convergence_rate:.6f}")

        # Append results to CSV
        fieldnames = ["Algorithm", "ChunkSize", "LearningRate", "BatchSize", "MAE", "MSE", "RMSE", "ConvergenceRate"]
        new_row = {
            "Algorithm": self.algo_name,
            "ChunkSize": self.chunk_size,
            "LearningRate": self.learning_rate,
            "BatchSize": self.batch_size,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "ConvergenceRate": convergence_rate
        }

        metrics_csv = f"metrics/metrics_summary.csv"

        os.makedirs("metrics", exist_ok=True)
        file_exists = os.path.isfile(metrics_csv)
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_row)

        # Plot the entire test
        plt.figure(figsize=(10, 5))
        plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
        plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
        plt.xlabel("Timestep")
        plt.ylabel("Speed (m/s)")
        plt.title(f"Test on full 1200-step dataset (chunk_size={self.chunk_size})")
        plt.legend()
        plt.tight_layout()

        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/{self.algo_name}_chunksize={self.chunk_size}_lr={str(self.learning_rate)}_bs={self.batch_size}.png")

    # ------------------------------------------------------------------------
    # Train a model
    # ------------------------------------------------------------------------
    def train(self):

        print(f"[INFO] Using algorithm: {self.algo_name}")
        print(f"[INFO] Using chunk_size = {self.chunk_size}")
        print(f"[INFO] Using learning_rate = {self.learning_rate}")
        print(f"[INFO] Using batch_size = {self.batch_size}")
        print(f"[INFO] Number of episodes: {len(self.episodes_list)} (some leftover if 1200 not divisible by {self.chunk_size})")

        # Name of model trained with the unique set of parameters
        self.trained_model_name = f"{self.algo_name}_lr={self.learning_rate}_bs={self.batch_size}_cs={self.chunk_size}_timesteps={self.total_timesteps}.zip"

        # Check if pre-trained model exists for the given set of parameters
        # If it doesn't exist, start training
        trained_model_path = f"{self.models_dir}/{self.trained_model_name}"
        if os.path.exists(f"{self.models_dir}/{self.trained_model_name}"):
            print(f"[INFO] {self.trained_model_name} already exists! Loading pretrained model")
            if self.algo_name == "SAC":
                self.model = SAC.load(trained_model_path)
            elif self.algo_name == "PPO":
                self.model = PPO.load(trained_model_path)
            elif self.algo_name == "TD3":
                self.model = TD3.load(trained_model_path)
            elif self.algo_name == "DDPG":
                self.model = DDPG.load(trained_model_path)
            else:
                raise ValueError("Invalid algorithm selected!")
        else:
            print(f"[INFO] {self.models_dir}/{self.trained_model_name} doesn't exist. Starting training")
            self.model = self.train_model(self.trained_model_name)

    def train_model(self, model_name):

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
        os.makedirs(self.models_dir, exist_ok=True)
        save_path = os.path.join(self.models_dir, model_name)
        self.model.save(save_path)
        print(f"[INFO] Model saved to: {save_path}")

        return self.model

# ------------------------------------------------------------------------
# Declare project tasks based on assignment document
# ------------------------------------------------------------------------
def task1():
    print("Task 1: Model and Hyperparameter Modifications")

    timesteps = 10_000

    # ------------------------------------------------------------------------
    # Try out different batch sizes
    # ------------------------------------------------------------------------
    algorithms = ["SAC", "PPO", "TD3", "DDPG"]
    batch_sizes = [128, 256, 512, 1024]

    # SAC, PPO, DDPG
    for algo in algorithms:
        for current_batch in batch_sizes:
            model_env = ModelEvaluationEnv(algo_name=algo, batch_size=current_batch, total_timesteps=timesteps) 
            model_env.train()

    # TD3
    batch_sizes = [64, 128, 256, 512, 1024]
    for current_batch in batch_sizes:
        model_env = ModelEvaluationEnv(algo_name="TD3", batch_size=current_batch, total_timesteps=timesteps) 
        model_env.train()

    # ------------------------------------------------------------------------
    # Try out different learning rates
    # ------------------------------------------------------------------------
    algorithms = ["SAC", "PPO", "DDPG"]
    learning_rates = [1e-3, 1e-4, 3e-4]

    for algo in algorithms:
        for current_lr in learning_rates:
            if algo != "PPO":
                model_env = ModelEvaluationEnv(algo_name=algo, learning_rate=current_lr, total_timesteps=timesteps)
            else:
                model_env = ModelEvaluationEnv(algo_name=algo, batch_size=64, learning_rate=current_lr, total_timesteps=timesteps)

            model_env.train()

    
def run_from_command_line(algo_name, batch_size, chunk_size, learning_rate, total_timesteps, log_dir):

    model_env = ModelEvaluationEnv(algo_name=algo_name, 
                                   batch_size=batch_size,
                                   learning_rate=learning_rate,
                                   chunk_size=chunk_size,
                                   total_timesteps=total_timesteps,
                                   log_dir=log_dir
                                   )

    # ------------------------------------------------------------------------
    # Train model based on given parameters
    # ------------------------------------------------------------------------
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
    model_env.compute_metrics(reference_speeds=reference_speeds, predicted_speeds=predicted_speeds, rewards=rewards)



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
        "--task",
        dest="task",
        type=str,
        default="cli",
        help="Select the task to run."
    )
    args = parser.parse_args()

    # Create logger
    log_dir = args.output_dir
    os.makedirs(log_dir, exist_ok=True)

    # Parse args
    algo_name = args.algorithm
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    
    learning_rate = args.learning_rate
    total_timesteps = args.timesteps
    task = args.task

    # Select task to run
    if (task == "cli"):
        run_from_command_line(algo_name, batch_size, chunk_size, learning_rate, total_timesteps, log_dir)
    elif task == "task1":
        task1()
    else:
        raise ValueError("Invalid task selected!")

if __name__ == "__main__":
    main()
