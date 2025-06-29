from stable_baselines3.common.callbacks import BaseCallback

class DrawdownStopCallback(BaseCallback):
    def __init__(self, max_drawdown, verbose=0):
        super().__init__(verbose)
        self.max_drawdown = max_drawdown
        self.peak = -float("inf")

    def _on_step(self):
        value = self.training_env.get_attr("balance")[0]
        if value > self.peak:
            self.peak = value
        drawdown = (self.peak - value) / (self.peak + 1e-8)
        if drawdown > self.max_drawdown:
            print(f"Early stopping: drawdown {drawdown:.2%} > {self.max_drawdown:.2%}")
            return False
        return True
