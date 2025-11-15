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

# Import functions for plotting data
from plot_data import *

# ------------------------------------------------------------------------
# Define ego vehicle constraints
# ------------------------------------------------------------------------

# Following distance constraints
MIN_DIST = 5.0
MAX_DIST = 30.0

# Acceleration constraints
MIN_ACCEL = -2.0
MAX_ACCEL = 2.0

# Jerk constraint
MAX_JERK = 1.0

# ------------------------------------------------------------------------
# Define ego vehicle training weights
# ------------------------------------------------------------------------

# Distance penalty
W_DIST = 2.0

# Speed penalty
W_SPEED = 1.0

# Jerk penalty for comfort
W_JERK = 0.2

# Weight to control how closly acceleration pushes stays against the min/max values
W_ACCEL_SOFT_PUSH = 0.05

def create_lead_vechicle(num_steps, csv_file="lead_vehicle_profile.csv"):

    # Generate lead vehicle speed profile
    speed = 10 + 5 * np.sin(0.02 * np.arange(num_steps)) + 2 * np.random.randn(num_steps)

    position = np.zeros(num_steps)
    for i in range(1, num_steps):
        position[i] = position[i-1] + speed[i-1]

    return speed, position

# ------------------------------------------------------------------------
# Always create a 1200-step speed dataset
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
# Create a 1200-step lead vehicle speed and position dataset
# ------------------------------------------------------------------------
lead_speed, lead_position = create_lead_vechicle(1200)

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

def select_algo(algo_name, train_env, device, learning_rate=1e-4, batch_size=256, sac_ent_coef="auto", ppo_ent_coef=0.005):

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
            batch_size=batch_size,
            buffer_size=200000,
            tau=0.005,
            gamma=0.99,
            ent_coef=sac_ent_coef,
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
            batch_size=batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ppo_ent_coef,
        )
    elif algo_name == "TD3":
        model = TD3(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=device,

            # Model specific params
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=1000000,
            tau=0.005,
            gamma=0.99,
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
            batch_size=batch_size,
            buffer_size=1000000,
            tau=0.005,
            gamma=0.99,
        )
    else:
        raise ValueError("Invalid algorithm selected!")

    return model

class BaseACCEnv(gym.Env):
    """
    Base class for ACC environments.

    Action:
        desired ego acceleration in [MIN_ACCEL, MAX_ACCEL] (m/s^2)
    Observation:
        [ego_speed, lead_speed, rel_distance_clamped>=0]
    """

    def __init__(self, delta_t=1.0):
        super().__init__()

        self.delta_t = delta_t

        # Action: desired acceleration with physical limits
        self.action_space = spaces.Box(
            low=MIN_ACCEL,
            high=MAX_ACCEL,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: [ego_speed, lead_speed, rel_distance>=0]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 50.0, 200.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Ego state
        self.step_idx = 0
        self.current_speed = 0.0
        self.ego_pos = 0.0
        self.last_accel = 0.0

        # Lead vehicle state (per episode)
        self.lead_speed = None  # np.array
        self.lead_pos = None    # np.array
        self.episode_len = 0

    # ------------------------------------------------------------------
    # Hooks to be implemented by subclasses
    # ------------------------------------------------------------------
    def _reset_lead_profile(self, seed=None):
        """
        Subclasses must:
            - set self.lead_speed (array shape [episode_len])
            - set self.lead_pos   (array shape [episode_len])
            - set self.episode_len (int)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------
    def compute_reward(self, gap, speed_diff, jerk, accel):
        """
        Reward based on:

          - distance to a target following distance (using raw signed gap)
          - speed difference (ego vs lead)
          - jerk (comfort)
          - soft penalty near accel limits
        """

        # Use a target following distance in the middle of [MIN_DIST, MAX_DIST]
        target_dist = 0.5 * (MIN_DIST + MAX_DIST)

        # gap is lead_pos - ego_pos (can be negative if ego is ahead)
        dist_error = gap - target_dist
        dist_pen = dist_error ** 2

        speed_pen = speed_diff * speed_diff
        jerk_pen = jerk * jerk
        acc_soft_pen = max(0.0, abs(accel) - 0.8 * MAX_ACCEL) ** 2

        reward = -(W_DIST * dist_pen + W_SPEED * speed_pen + W_JERK * jerk_pen + W_ACCEL_SOFT_PUSH * acc_soft_pen)

        # Normalization
        reward /= 1000.0

        return reward

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Subclass defines lead profile and episode length
        self._reset_lead_profile(seed)

        # Reset ego state
        self.step_idx = 0
        self.current_speed = 0.0
        self.ego_pos = 0.0
        self.last_accel = 0.0

        # Initial observation
        gap = self.lead_pos[0] - self.ego_pos               # raw (can be negative)
        rel_distance = max(gap, 0.0)                         # clamped for obs
        lead_speed = self.lead_speed[0]

        obs = np.array(
            [self.current_speed, lead_speed, rel_distance],
            dtype=np.float32,
        )
        info = {}
        return obs, info

    def step(self, action):
        # --------------------------------------------------------------
        # 1) Process action: clip + jerk limit
        # --------------------------------------------------------------
        desired_accel = float(np.clip(action[0], MIN_ACCEL, MAX_ACCEL))
        max_accel_delta = MAX_JERK * self.delta_t

        accel_limited = np.clip(
            desired_accel,
            self.last_accel - max_accel_delta,
            self.last_accel + max_accel_delta,
        )
        ego_accel = float(np.clip(accel_limited, MIN_ACCEL, MAX_ACCEL))

        jerk = (ego_accel - self.last_accel) / self.delta_t
        self.last_accel = ego_accel

        # --------------------------------------------------------------
        # 2) Integrate ego dynamics
        # --------------------------------------------------------------
        self.current_speed = max(self.current_speed + ego_accel * self.delta_t, 0.0)
        self.ego_pos += self.current_speed * self.delta_t

        # --------------------------------------------------------------
        # 3) Lead / relative states at current index
        # --------------------------------------------------------------
        idx = self.step_idx
        lead_speed = self.lead_speed[idx]
        gap = self.lead_pos[idx] - self.ego_pos             # raw gap (can be negative)
        rel_distance = max(gap, 0.0)                        # clamped for obs/info
        speed_diff = self.current_speed - lead_speed

        reward = self.compute_reward(gap, speed_diff, jerk, ego_accel)

        # Lead acceleration (finite difference)
        if idx == 0:
            lead_accel = 0.0
        else:
            lead_accel = (self.lead_speed[idx] - self.lead_speed[idx - 1]) / self.delta_t

        acc_diff = ego_accel - lead_accel

        # --------------------------------------------------------------
        # 4) Next observation & termination
        # --------------------------------------------------------------
        next_idx = idx + 1
        terminated = next_idx >= self.episode_len
        obs_index = self.episode_len - 1 if terminated else next_idx

        next_lead_speed = self.lead_speed[obs_index]
        next_gap = self.lead_pos[obs_index] - self.ego_pos
        next_rel_distance = max(next_gap, 0.0)

        obs = np.array(
            [self.current_speed, next_lead_speed, next_rel_distance],
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # 5) Info dict (before advancing index)
        # --------------------------------------------------------------
        info = {
            "speed_error": abs(speed_diff),
            "rel_distance": rel_distance,   # clamped
            "gap_raw": gap,                 # optional: raw signed gap
            "speed_diff": speed_diff,
            "jerk": jerk,
            "lead_speed": lead_speed,
            "lead_pos": self.lead_pos[idx],
            "ego_speed": self.current_speed,
            "ego_pos": self.ego_pos,
            "ego_accel": ego_accel,
            "lead_accel": lead_accel,
            "acc_diff": acc_diff,
        }

        # --------------------------------------------------------------
        # 6) Advance time index
        # --------------------------------------------------------------
        self.step_idx = next_idx
        truncated = False

        return obs, reward, terminated, truncated, info

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
# ------------------------------------------------------------------------
class TrainEnv(BaseACCEnv):
    """
    Training environment: each reset picks a random chunk from episodes_list
    and creates a fresh lead vehicle profile for that episode.
    """
    def __init__(self, episodes_list, delta_t=1.0):
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.current_episode = None
        super().__init__(delta_t=delta_t)

    def _reset_lead_profile(self, seed=None):
        # Randomly select an episode/chunk
        episode_index = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[episode_index]
        self.episode_len = len(self.current_episode)

        # Fresh lead profile for this episode (promotes generalization)
        self.lead_speed, self.lead_pos = create_lead_vechicle(self.episode_len)


# ------------------------------------------------------------------------
# 3B) Test Environment with same reward flexibility
# ------------------------------------------------------------------------
class TestEnv(BaseACCEnv):
    """
    Test environment: uses a deterministic lead vehicle profile over the full
    dataset for reproducible evaluation.
    """
    def __init__(self, full_data, delta_t=1.0):
        self.full_data = full_data
        self.n_steps = len(full_data)

        # Precompute deterministic lead profile (same as before)
        rng = np.random.RandomState(42)
        t = np.arange(self.n_steps)
        noise = rng.normal(0, 0.5, size=self.n_steps)
        lead_speed = np.clip(12.0 + 4.0 * np.sin(0.02 * t) + noise, 0.0, None)

        lead_pos = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            lead_pos[i] = lead_pos[i - 1] + lead_speed[i - 1] * delta_t

        # Store profiles; they'll be used every reset
        self._lead_speed_profile = lead_speed
        self._lead_pos_profile = lead_pos

        super().__init__(delta_t=delta_t)

    def _reset_lead_profile(self, seed=None):
        # Reuse the same deterministic profile each episode
        self.lead_speed = self._lead_speed_profile
        self.lead_pos = self._lead_pos_profile
        self.episode_len = self.n_steps


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
        with open(self.log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestep", "average_reward", "episode_return"])

    def _on_step(self) -> bool:
        t = self.num_timesteps
        r = self.locals.get("rewards", [0])[-1]
        self.episode_rewards.append(r)

        done = self.locals.get("dones", [False])[-1]
        if done:
            avg_reward = float(np.mean(self.episode_rewards))
            episode_return = float(np.sum(self.episode_rewards))

            with open(self.log_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward, episode_return])

            self.logger.record("reward/average_reward", avg_reward)
            self.logger.record("reward/episode_return", episode_return)
            self.episode_rewards.clear()

        return True

class ModelEvaluationEnv():

    def __init__(self, log_dir="logs", algo_name="SAC", learning_rate=3e-4, batch_size=256, sac_ent_coef="auto", ppo_ent_coef=0.005, total_timesteps=100_000, episode_len=100, models_dir="trained_models"):
        self.algo_name = algo_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sac_ent_coef = sac_ent_coef
        self.ppo_ent_coef = ppo_ent_coef

        if self.algo_name == "SAC":
            self.ent_coef = sac_ent_coef
        elif self.algo_name == "PPO":
            self.ent_coef = ppo_ent_coef
        else:
            self.ent_coef = None

        self.episode_len = episode_len
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.models_dir = models_dir
        self.logger = configure(self.log_dir, ["stdout", "tensorboard"])
        self.trained_model_name = None

        self.episodes_list = chunk_into_episodes(full_speed_data, self.episode_len)

        self.model = None

    def compute_metrics(self, lead_speeds, ego_speeds, ego_jerks, rewards, avg_reward, filename):
        """
        Compute performance metrics after testing.
        """

        lead = np.array(lead_speeds)
        ego = np.array(ego_speeds)

        mae = mean_absolute_error(lead, ego)
        mse = mean_squared_error(lead, ego)
        rmse = np.sqrt(mse)

        # Convergence rate (very rough): average reward improvement per 1k steps
        # You can refine this later if you log intermediate rewards.
        convergence_rate = np.mean(rewards[-100:]) - np.mean(rewards[:100])
        convergence_rate /= self.total_timesteps / 1000  # normalize per 1k steps

        # Calculate mean and variance of jerk        
        avg_jerk = np.mean(np.abs(ego_jerks))
        var_jerk = np.var(ego_jerks)

        print(f"[METRICS] MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, AvgReward={avg_reward:.4f}, ConvergenceRate={convergence_rate:.6f}, AvgJerk={avg_jerk:.4f}, VarianceJerk={var_jerk:.4f}")

        # Append results to CSV
        fieldnames = ["Algorithm", "EpisodeLength", "LearningRate", "BatchSize", "EntCoef", "MAE", "MSE", "RMSE", "AvgReward", "ConvergenceRate", "AvgJerk", "VarianceJerk"]
        new_row = {
            "Algorithm": self.algo_name,
            "EpisodeLength": self.episode_len,
            "LearningRate": self.learning_rate,
            "BatchSize": self.batch_size,
            "EntCoef": self.ent_coef,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "AvgReward": avg_reward,
            "ConvergenceRate": convergence_rate,
            "AvgJerk": avg_jerk, 
            "VarianceJerk": var_jerk,
        }

        metrics_csv = f"metrics/{filename}.csv"

        os.makedirs("metrics", exist_ok=True)
        file_exists = os.path.isfile(metrics_csv)
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_row)

        # Plot the entire test
        plt.figure(figsize=(10, 5))
        plt.plot(lead_speeds, label="Lead Speed", linestyle="--")
        plt.plot(ego_speeds, label="Ego Speed", linestyle="-")
        plt.xlabel("Timestep")
        plt.ylabel("Speed (m/s)")
        plt.title(f"Test on full 1200-step dataset (episode_len={self.episode_len})")
        plt.legend()
        plt.tight_layout()

        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/{self.algo_name}_lr={self.learning_rate}_bs={self.batch_size}_el={self.episode_len}_entcoef={self.ent_coef}_timesteps={self.total_timesteps}.png")

    # ------------------------------------------------------------------------
    # Train a model
    # ------------------------------------------------------------------------
    def train(self):

        # Name of model trained with the unique set of parameters
        self.trained_model_name = f"{self.algo_name}_lr={self.learning_rate}_bs={self.batch_size}_el={self.episode_len}_entcoef={self.ent_coef}_timesteps={self.total_timesteps}.zip"

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

        print(f"[INFO] Using algorithm: {self.algo_name}")
        print(f"[INFO] Using episode_len = {self.episode_len}")
        print(f"[INFO] Using learning_rate = {self.learning_rate}")
        print(f"[INFO] Using batch_size = {self.batch_size}")
        print(f"[INFO] Using entropy coefficient = {self.ent_coef}")
        print(f"[INFO] Number of episodes: {len(self.episodes_list)} (some leftover if 1200 not divisible by {self.episode_len})")

        # 5B) Create the TRAIN environment
        def make_train_env():
            return TrainEnv(self.episodes_list, delta_t=1.0)

        train_env = DummyVecEnv([make_train_env])

        # 5C) Build the model (SAC with MlpPolicy)
        device = torch.device("cuda" if torch.cuda.is_available() and self.algo_name != "PPO" else "cpu")
        print(f"Training on device: {device}")

        # Select the algorithm and associated hyperparameters
        self.model = select_algo(
            self.algo_name,
            train_env,
            device,
            self.learning_rate,
            self.batch_size,
            sac_ent_coef=self.sac_ent_coef,
            ppo_ent_coef=self.ppo_ent_coef,
        )
        self.model.set_logger(self.logger)

        callback = CustomLoggingCallback(self.log_dir, f"{self.trained_model_name[:-4]}_training_log.csv")

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

    def test(self, metrics_summary_filename):
        test_env = TestEnv(full_speed_data, delta_t=1.0)

        obs, _ = test_env.reset()
        rewards = []

        ego_accels, ego_speeds, ego_positions = [], [], []
        lead_accels, lead_speeds, lead_positions = [], [], []
        acc_diffs = []
        rel_distances = []
        jerks = []
                

        for _ in range(DATA_LEN):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            rewards.append(reward)

            ego_speeds.append(info["ego_speed"])
            ego_positions.append(info["ego_pos"])
            lead_speeds.append(info["lead_speed"])
            lead_positions.append(info["lead_pos"])
            rel_distances.append(info["rel_distance"])
            ego_accels.append(info["ego_accel"])
            lead_accels.append(info["lead_accel"])
            acc_diffs.append(info["acc_diff"])
            jerks.append(info["jerk"])

            if terminated or truncated:
                break

        avg_test_reward = float(np.mean(rewards))
        print(f"[TEST] Average reward over 1200-step test: {avg_test_reward:.3f}")

        # Calculate metrics
        self.compute_metrics(
            lead_speeds=lead_speeds,
            ego_speeds=ego_speeds,
            ego_jerks=jerks,
            rewards=rewards,
            avg_reward=avg_test_reward,
            filename=metrics_summary_filename,
        )

        # write the timeseries CSV
        os.makedirs("metrics", exist_ok=True)

        out_name = self.trained_model_name[:-4]
        data_csv = os.path.join("metrics", f"lead_vs_ego_data_{out_name}.csv")

        pd.DataFrame({
            "lead_speed": lead_speeds,
            "ego_speed": ego_speeds,
            "lead_pos": lead_positions,
            "ego_pos": ego_positions,
            "rel_distance": rel_distances,
            "jerk": jerks,
            "ego_accel": ego_accels,
            "lead_accel": lead_accels,
            "acc_diff": acc_diffs,
            "reward": rewards[:len(ego_speeds)],
        }).to_csv(data_csv, index=False)

        print(f"[INFO] Wrote timeseries to {data_csv}")

        plot_lead_vs_ego(data_csv, out_name=f"{out_name}", algo=self.algo_name)
        plot_position_difference(data_csv, out_name=f"{out_name}", algo=self.algo_name)
        plot_speed_difference(data_csv, out_name=f"{out_name}", algo=self.algo_name)
        plot_acceleration_difference(data_csv, out_name=f"{out_name}", algo=self.algo_name, band=MAX_ACCEL)
        plot_training_curve(
            csv_path=f"logs/{out_name}_training_log.csv",
            out_name=f"{out_name}",
            save_dir="images",
            smooth_window=self.episode_len,
        )

# ------------------------------------------------------------------------
# Define number of training timesteps for each model
# ------------------------------------------------------------------------
timesteps = {
    "SAC": 10_000,
    "PPO": 50_000,
    "TD3": 50_000,
    "DDPG": 50_000,
}

# ------------------------------------------------------------------------
# Declare project tasks based on assignment document
# ------------------------------------------------------------------------
def batch_size_test():
    print("------------------------------------------------------------------------")
    print(" Batch size modification (64, 128, 256) ")
    print("------------------------------------------------------------------------")

    ep_len = {
        "SAC": 50,
        "PPO": 100,
        "TD3": 100,
        "DDPG": 100,
    }

    lr = {
        "SAC": 50,
        "PPO": 100,
        "TD3": 100,
        "DDPG": 100,
    }

    algorithms = ["SAC", "PPO", "TD3", "DDPG"]
    batch_sizes = [64, 128, 256]
    metrics_summary_filename = "metrics_summary_task1_batchsize_variation"

    for algo in algorithms:
        for current_batch in batch_sizes:
            model_env = ModelEvaluationEnv(algo_name=algo, batch_size=current_batch, learning_rate=1e-4, episode_len=ep_len[algo], total_timesteps=timesteps[algo]) 
            model_env.train()
            model_env.test(metrics_summary_filename)

    plot_batchsize_vs_metric(csv_path=f"metrics/{metrics_summary_filename}.csv", metric="MAE", out_name="task1_batchsize_vs_MAE", save_dir="task1_images")

def lr_test():

    print("------------------------------------------------------------------------")
    print(" Learning rate modification (1e-4, 3e-4, 1e-3) ")
    print("------------------------------------------------------------------------")

    batch_size = {
        "SAC": 256,
        "PPO": 64,
        "TD3": 64,
        "DDPG": 128,
    }

    ep_len = {
        "SAC": 50,
        "PPO": 100,
        "TD3": 100,
        "DDPG": 100,
    }

    algorithms = ["SAC", "PPO", "TD3", "DDPG"]
    learning_rates = [1e-4, 3e-4, 1e-3]
    metrics_summary_filename = "metrics_summary_task1_learningrate_variation"

    for algo in algorithms:
        for current_lr in learning_rates:
            model_env = ModelEvaluationEnv(algo_name=algo, batch_size=batch_size[algo], episode_len=ep_len[algo], learning_rate=current_lr, total_timesteps=timesteps[algo])
            model_env.train()
            model_env.test(metrics_summary_filename)

    plot_learningrate_vs_metric(csv_path=f"metrics/{metrics_summary_filename}.csv", metric="MAE", out_name="task1_lr_vs_MAE", save_dir="task1_images")

def ent_coef_test():

    print("------------------------------------------------------------------------")
    print(" Entropy coefficient modification (for SAC and PPO) ")
    print("------------------------------------------------------------------------")

    metrics_summary_filename = "metrics_summary_task1_ent_coefficient_variation"

    # SAC
    ent_coefficients = ["auto", 0.0, 0.01]
    for ent in ent_coefficients:
        model_env = ModelEvaluationEnv(algo_name="SAC", batch_size=256, learning_rate=1e-4, episode_len=50, total_timesteps=timesteps["SAC"], sac_ent_coef=ent)
        model_env.train()
        model_env.test(metrics_summary_filename)

    plot_entropy_vs_metric(csv_path=f"metrics/{metrics_summary_filename}.csv", algo="SAC", metric="MAE", out_name="task1_SAC_entropy_vs_MAE", save_dir="task1_images")

    # PPO
    ent_coefficients = [0.005, 0.01, 0.05]
    for ent in ent_coefficients:
        model_env = ModelEvaluationEnv(algo_name="PPO", batch_size=64, learning_rate=3e-4, episode_len=100, total_timesteps=timesteps["PPO"], ppo_ent_coef=ent)
        model_env.train()
        model_env.test(metrics_summary_filename)

    plot_entropy_vs_metric(csv_path=f"metrics/{metrics_summary_filename}.csv", algo="PPO", metric="MAE", out_name="task1_PPO_entropy_vs_MAE", save_dir="task1_images")

def episode_test():
    print("------------------------------------------------------------------------")
    print(" Episode length modification (50, 100, 200) ")
    print("------------------------------------------------------------------------")

    algorithms = ["SAC", "PPO", "TD3", "DDPG"]
    episode_lengths = [50, 100, 200]
    metrics_summary_filename = "metrics_summary_task2_episodelength_variation"

    for algo in algorithms:
        for length in episode_lengths:
            print("------------------------------------------------------------------------")
            print(f" Episode length modification | algo = {algo} | episode_len = {length} ")
            print("------------------------------------------------------------------------")
            model_env = ModelEvaluationEnv(algo_name=algo, total_timesteps=timesteps[algo], episode_len=length, ppo_ent_coef=0.005, sac_ent_coef=0.01) 
            model_env.train()
            model_env.test(metrics_summary_filename)

    plot_episodelength_vs_metric(csv_path=f"metrics/{metrics_summary_filename}.csv", metric="MAE", out_name="task2_episodelength_vs_MAE", save_dir="task2_images")
    plot_episodelength_vs_metric(csv_path=f"metrics/{metrics_summary_filename}.csv", metric="RMSE", out_name="task2_episodelength_vs_RMSE", save_dir="task2_images")
    plot_episodelength_vs_metric(csv_path=f"metrics/{metrics_summary_filename}.csv", metric="ConvergenceRate", out_name="task2_episodelength_vs_ConvergenceRate", save_dir="task2_images")

def best_params_test():

    print("------------------------------------------------------------------------")
    print(" Collect data on the best parameters for each model                     ")
    print("------------------------------------------------------------------------")

    metrics_summary_filename = "metrics_summary_task1_best_hyperparameters"

    # SAC
    model_env = ModelEvaluationEnv(algo_name="SAC", learning_rate=1e-4, batch_size=256, sac_ent_coef=0.01, total_timesteps=timesteps["SAC"])
    model_env.train()
    model_env.test(metrics_summary_filename)

    # PPO
    model_env = ModelEvaluationEnv(algo_name="PPO", learning_rate=3e-4, batch_size=64, ppo_ent_coef=0.005, total_timesteps=timesteps["PPO"])
    model_env.train()
    model_env.test(metrics_summary_filename)

    # TD3
    model_env = ModelEvaluationEnv(algo_name="TD3", learning_rate=1e-4, batch_size=128, total_timesteps=timesteps["TD3"])
    model_env.train()
    model_env.test(metrics_summary_filename)

    # DDPG
    model_env = ModelEvaluationEnv(algo_name="DDPG", learning_rate=1e-4, batch_size=64, total_timesteps=timesteps["DDPG"])
    model_env.train()
    model_env.test(metrics_summary_filename)


def run_from_command_line(algo_name, batch_size, episode_len, learning_rate, total_timesteps, log_dir):

    model_env = ModelEvaluationEnv(algo_name=algo_name, 
                                   batch_size=batch_size,
                                   learning_rate=learning_rate,
                                   episode_len=episode_len,
                                   total_timesteps=total_timesteps,
                                   log_dir=log_dir
                                   )

    # ------------------------------------------------------------------------
    # Train and test model
    # ------------------------------------------------------------------------
    metrics_filename = "cli_metrics"
    model_env.train()
    model_env.test(metrics_filename)

# ------------------------------------------------------------------------
# 5) Main: user sets episode_len from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--episode_len",
        type=int,
        default=100,
        help="Episode length for training."
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
    episode_len = args.episode_len
    
    learning_rate = args.learning_rate
    total_timesteps = args.timesteps
    task = args.task

    # Select task to run
    if (task == "cli"):
        run_from_command_line(algo_name, batch_size, episode_len, learning_rate, total_timesteps, log_dir)
    elif task == "batch_size_test":
        batch_size_test()
    elif task == "lr_test":
        lr_test()
    elif task == "ent_coef_test":
        ent_coef_test()
    elif task == "best_params_test":
        best_params_test()
    elif task == "episode_test":
        episode_test()
    else:
        raise ValueError("Invalid task selected!")

if __name__ == "__main__":
    main()
