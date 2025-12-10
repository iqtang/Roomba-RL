from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RoombaEvalCallback(BaseCallback):
    def __init__(self, verbose=0, print_every=2):
        super().__init__(verbose)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_cells = []
        self.episode_collisions = []
        self.episode_coverage = []

        self.losses = []

        self.current_reward = 0
        self.current_length = 0
        self.current_cells = 0
        self.current_collisions = 0
        self.current_coverage = 0.0

        self.print_every = print_every

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]

        self.current_reward += reward
        self.current_length += 1
        if info.get("new_cell", False):
            self.current_cells += 1
        if info.get("collision", False):
            self.current_collisions += 1
        self.current_coverage = info.get("coverage", self.current_coverage)

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_cells.append(self.current_cells)
            self.episode_collisions.append(self.current_collisions)
            self.episode_coverage.append(self.current_coverage)


            if len(self.episode_rewards) % self.print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-self.print_every:])
                avg_coverage = np.mean(self.episode_coverage[-self.print_every:])
                print(f"Episode {len(self.episode_rewards)} - avg reward last {self.print_every}: {avg_reward:.2f}, "
                      f"avg coverage: {avg_coverage:.2f}")

            self.current_reward = 0
            self.current_length = 0
            self.current_cells = 0
            self.current_collisions = 0
            self.current_coverage = 0.0

        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout collection, can access loss stats"""
        logger_dict = self.model.logger.name_to_value
        if "train/policy_loss" in logger_dict:
            policy_loss = logger_dict["train/policy_loss"]
            value_loss = logger_dict.get("train/value_loss", 0)
            self.losses.append((policy_loss, value_loss))
