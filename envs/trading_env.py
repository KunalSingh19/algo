import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Single-asset or multi-asset trading environment.
    """
    def __init__(self, df, window_size=50, max_drawdown=0.3, commission=0.001, slippage=0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.max_drawdown = max_drawdown
        self.commission = commission
        self.slippage = slippage
        self.current_step = window_size
        self.balance = 1.0
        self.positions = np.zeros(self.n_assets)
        self.max_portfolio_value = self.balance
        self.reset()

    @property
    def n_assets(self):
        return len([c for c in self.df.columns if c.endswith('Close')])

    def reset(self, *, seed=None, options=None):
        self.current_step = self.window_size
        self.balance = 1.0
        self.positions = np.zeros(self.n_assets)
        self.max_portfolio_value = self.balance
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Action: vector of weights (sums to 1)
        # Simulate next step, update positions, calculate reward
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        self.current_step += 1
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        # Return most recent window of features
        return self.df.iloc[self.current_step - self.window_size:self.current_step].values.flatten()
