from .reward_base import RewardBase
from .sharpe_reward import SharpeReward
from .drawdown_penalty import DrawdownPenalty

class CustomReward(RewardBase):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.sharpe = SharpeReward()
        self.drawdown = DrawdownPenalty()
        self.alpha = alpha
        self.beta = beta

    def compute(self, returns, portfolio_values):
        return self.alpha * self.sharpe.compute(returns) + self.beta * self.drawdown.compute(portfolio_values)
