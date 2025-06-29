# sharpe_reward.py 
import numpy as np
from .reward_base import RewardBase

class SharpeReward(RewardBase):
    def compute(self, returns):
        if len(returns) < 2:
            return 0
        excess = np.array(returns) - 0
        return np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252)
