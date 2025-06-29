# RL Meta-Ensemble Portfolio Framework

## Overview

A modular, research-friendly reinforcement learning pipeline for multi-asset trading and portfolio optimization.  
Includes custom environments, technical features, reward engineering, and SB3 agent training out-of-the-box.

## Structure

- `config/`: All experiment, agent, and environment configs
- `data/`: Data download, preprocessing, splitting
- `envs/`: Custom Gym environments (multi-asset, wrappers)
- `features/`: Technical indicators, alpha signals
- `models/`: RL agent logic, training, evaluation, callbacks
- `rewards/`: Reward classes and custom reward logic
- `utils/`: Logging, plotting, metrics
- `tests/`: Unit tests for every major component

## Usage

```bash
pip install -r requirements.txt
python main.py
