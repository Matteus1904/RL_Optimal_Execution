from .util import OptimizerSampled, DataBuffer, EpisodicSampler                                                                                                                     
import torch
from torch import nn
from typing import Dict

class Critic:
    def __init__(
        self,
        td_n: int,
        discount_factor: float,
        device: str,
        model: nn.Module,
        optimizer: OptimizerSampled,
    ):
        """Instantiate Critic

        :param td_n: number of terms in temporal difference objective
        :type td_n: int
        :param discount_factor: discount factor to use in temproal difference objective
        :type discount_factor: float
        :param device: device for model fitting
        :type device: str
        :param model: NN network that should fit the Value function
        :type model: nn.Module
        :param optimizer: optimizer for fitting of Value function
        :type optimizer: Optimizer
        """
        self.model = model
        self.td_n = td_n
        self.device = device
        self.discount_factor = discount_factor
        self.optimizer = optimizer

    def objective(self, batch: Dict[str, torch.FloatTensor]) -> torch.FloatTensor:
        """Calculate temporal difference objective

        :param batch: dict that contains episodic data: observations, running_costs
        :type batch: Dict[str, torch.FloatTensor]
        :return: temporal difference objective
        :rtype: torch.FloatTensor
        """

        observations = batch["observation"]
        running_costs = batch["running_cost"]

        J = self.model(observations)
        N = observations.shape[0]
        objective_temp = J - self.discount_factor ** self.td_n * J.roll(-self.td_n)
        for i in range(self.td_n):
             objective_temp -= self.discount_factor ** i * running_costs.roll(-i)
        objective_temp_2 = torch.linalg.norm(objective_temp[:-self.td_n]) ** 2
        objective = objective_temp_2 / ( N - 1 - self.td_n )
        return objective

    def fit(self, buffer: DataBuffer) -> None:
        """Runs optimization procedure for critic

        :param buffer: data buffer with experience replay
        :type buffer: DataBuffer
        """
        self.model.to(self.device)
        self.optimizer.optimize(
            self.objective,
            buffer.iter_batches(
                batch_sampler=EpisodicSampler,
                keys=[
                    "observation",
                    "observation_action",
                    "running_cost",
                ],
                device=self.device,
            ),
        )