#!/bin/bash

# SAC
time ./rl.py -a SAC -lr 3e-4 -bs 128
time ./rl.py -a SAC -lr 3e-4 -bs 256
time ./rl.py -a SAC -lr 3e-4 -bs 512
time ./rl.py -a SAC -lr 3e-4 -bs 1024

time ./rl.py -a SAC -lr 1e-3 -bs 256
time ./rl.py -a SAC -lr 1e-4 -bs 256
time ./rl.py -a SAC -lr 3e-4 -bs 256

# PPO
time ./rl.py -a PPO -lr 3e-4 -bs 64
time ./rl.py -a PPO -lr 3e-4 -bs 128
time ./rl.py -a PPO -lr 3e-4 -bs 256
time ./rl.py -a PPO -lr 3e-4 -bs 512

time ./rl.py -a PPO -lr 1e-3 -bs 64
time ./rl.py -a PPO -lr 1e-4 -bs 64
time ./rl.py -a PPO -lr 3e-4 -bs 64

# TD3
time ./rl.py -a TD3 -lr 3e-4 -bs 128
time ./rl.py -a TD3 -lr 3e-4 -bs 256
time ./rl.py -a TD3 -lr 3e-4 -bs 512
time ./rl.py -a TD3 -lr 3e-4 -bs 1024

time ./rl.py -a TD3 -lr 1e-3 -bs 256
time ./rl.py -a TD3 -lr 1e-4 -bs 256
time ./rl.py -a TD3 -lr 3e-4 -bs 256

# DDPG
time ./rl.py -a DDPG -lr 3e-4 -bs 128
time ./rl.py -a DDPG -lr 3e-4 -bs 256
time ./rl.py -a DDPG -lr 3e-4 -bs 512
time ./rl.py -a DDPG -lr 3e-4 -bs 1024

time ./rl.py -a DDPG -lr 1e-3 -bs 256
time ./rl.py -a DDPG -lr 1e-4 -bs 256
time ./rl.py -a DDPG -lr 3e-4 -bs 256
