# RL-Meta-Ensemble

A modular, production-grade reinforcement learning portfolio optimizer using PPO (LSTM), Stable-Baselines3, and custom feature/reward engineering.

---

## Features

- Multi-asset RL trading via PPO (LSTM)
- Custom OpenAI Gym environment
- Feature engineering: returns, indicators, volume, correlations, regime detection
- Risk-adjusted, configurable reward functions
- Training, evaluation, and logging pipeline
- Backtesting, plotting, and metrics dashboard
- Modular structure for research and deployment

---

## Structure

```
RL-Meta-Ensemble/
├── config/
├── data/
├── envs/
├── features/
├── models/
├── rewards/
├── utils/
├── main.py
```

---

## Quickstart

1. Install requirements:  
   `pip install -r requirements.txt`

2. Run:  
   `python main.py --config config/experiment.yaml`

3. View results in `models/` and dashboards in `utils/plotting.py`

---

## See each folder for details.
