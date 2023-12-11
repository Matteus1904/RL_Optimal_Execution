from .util import OptimizerSampled, DataBuffer, RollingBatchSampler
from .critic import Critic
import torch
from torch import nn
from typing import Dict
import numpy as np
    
class Policy:
    def __init__(
        self,
        model: nn.Module,
        optimizer: OptimizerSampled,
        discount_factor: float,
        critic: Critic,
        device: str = "cpu",
    ) -> None:
        """Initialize policy

        Args:
            model (nn.Module): model to optimize
            optimizer (Optimizer): optimizer for `model` weights optimization
            device (str, optional): device for gradient descent optimization procedure. Defaults to "cpu".
            discount_factor (float): discount factor gamma for running costs
            critic (Critic): Critic class that contains model for Value function
        """
        self.discount_factor = discount_factor
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.critic = critic

    def objective(self, batch: Dict["str", torch.tensor]) -> torch.tensor:
        """This method computes a proxy objective specifically for automatic differentiation since its gradient is exactly as in REINFORCE

        Args:
            batch (torch.tensor): batch with observations, actions, step_ids, episode_ids, running costs

        Returns:
            torch.tensor: objective value
        """

        observations = batch["observation"]
        actions = batch["action"]
        step_ids = batch["step_id"]
        episode_ids = batch["episode_id"].type(torch.int64)
        running_costs = batch["running_cost"]
        N_episodes = self.N_episodes
        log_probs = self.model.log_probs(observations, actions).reshape(-1, 1)
        critic_values = self.critic.model(observations).detach()

        step_ids = step_ids.type(torch.int64)
        steps_total = int(step_ids.max()) + 1

        running_costs = running_costs.reshape(N_episodes, steps_total)
        critic_values = critic_values.reshape(N_episodes, steps_total)
        log_probs = log_probs.reshape(N_episodes, steps_total)
        objective_temp = torch.pow(self.discount_factor,torch.arange(0,steps_total,dtype=int)).repeat((N_episodes,1))
        objective_temp_2 = torch.sum((objective_temp * ( running_costs + self.discount_factor * critic_values.roll(-1) - critic_values ) * log_probs)[:,:-1])
        objective = objective_temp_2 / N_episodes
        return objective

    def get_N_episodes(self, buffer: DataBuffer):
        return len(np.unique(buffer.data["episode_id"]))

    def REINFORCE_step(self, buffer: DataBuffer) -> None:
        """Do gradient REINFORCE step"""
        self.N_episodes = self.get_N_episodes(buffer)
        self.model.to(self.device)
        self.optimizer.optimize(
            self.objective,
            buffer.iter_batches(
                keys=[
                    "observation",
                    "observation_action",
                    "action",
                    "running_cost",
                    "episode_id",
                    "step_id",
                ],
                batch_sampler=RollingBatchSampler,
                mode="full",
                n_batches=1,
            ),
        )
        self.model.to("cpu")
        buffer.nullify_buffer()
