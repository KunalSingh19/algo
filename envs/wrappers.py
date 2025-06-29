# Example: Observation normalization wrapper
import gymnasium as gym

class ObsNormWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_mean = 0
        self.obs_std = 1

    def observation(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
