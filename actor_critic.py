import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
from tqdm import tqdm
from typing import Callable
import yaml

from lib.system import MarketSystem
from lib.simulator import EM_Simulator
from lib.util import DataBuffer
from lib.util import OptimizerSampled
from lib.model import ModelPerceptron
from lib.critic import Critic
from lib.model import GaussianPDFModelCus
from lib.policy import Policy

import argparse
# Instantiate the parser

class MonteCarloSimulationScenario:
    """Run whole REINFORCE procedure"""

    def __init__(
        self,
        train_mode: True,
        simulator: EM_Simulator,
        system: MarketSystem,
        policy: Policy,
        critic: Critic,
        N_episodes: int,
        N_iterations: int,
        discount_factor: float = 1.0,
        termination_criterion: Callable[
            [np.array, np.array, float, float], bool
        ] = lambda *args: False,
    ):
        """Initialize scenario for main loop


        Args:
            simulator (Simulator): simulator for computing system dynamics
            system (MarketSystem): system itself
            policy (PolicyREINFORCE): REINFORCE gradient stepper
            N_episodes (int): number of episodes in one iteration
            N_iterations (int): number of iterations
            discount_factor (float, optional): discount factor for running costs. Defaults to 1
            termination_criterion (Callable[[np.array, np.array, float, float], bool], optional): criterion for episode termination. Takes observation, action, running_cost, total_cost. Defaults to lambda*args:False
        """
        self.train_mode = train_mode
        self.simulator = simulator
        self.system = system
        self.policy = policy
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.termination_criterion = termination_criterion
        self.discount_factor = discount_factor
        self.data_buffer = DataBuffer()
        self.critic = critic
        self.total_cost = 0
        self.total_costs_episodic = []
        self.learning_curve = []
        self.last_observations = None
        self.toe = []
        self.twap = []

    def compute_running_cost(
        self, observation: np.array, action: np.array
    ) -> float:
        """Computes running cost

        Args:
            observation (np.array): current observation
            action (np.array): current action

        Returns:
            float: running cost value
        """

        return -(observation[1] -0.001*action[0]) * action[0] +(observation[0] <0)

    def run(self) -> None:
        """Run main loop"""

        eps = 0.1
        means_total_costs = [eps]
        self.toe = []
        self.twap = []
        for iteration_idx in range(self.N_iterations):
            toe = []
            twap = []
            for episode_idx in tqdm(range(self.N_episodes)):
                toe1 = []
                twap1 = []
                terminated = False
                while self.simulator.step():
                    (
                        observation,
                        action,
                        step_idx,
                    ) = self.simulator.get_sim_step_data()
                    new_action = (
                        self.policy.model.sample(torch.tensor(observation[:,0]).float())
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    running_cost = self.compute_running_cost(
                        observation[:,0], new_action
                    )
                    discounted_running_cost = (
                        self.discount_factor ** (step_idx) * running_cost
                    )
                    self.total_cost += discounted_running_cost

                    if not terminated and self.termination_criterion(
                        observation[:,0],
                        new_action,
                        discounted_running_cost,
                        self.total_cost,
                    ):
                        terminated = True

                    if not terminated:
                        self.data_buffer.push_to_end(
                            observation=np.copy(observation[:,0]),
                            action=np.copy(new_action),
                            observation_action=np.hstack((observation[:,0], action[:1])),
                            running_cost=np.copy(running_cost),
                            discounted_running_cost=np.copy(
                                discounted_running_cost
                            ),
                            total_cost=np.copy(self.total_cost),
                            step_id=step_idx - 1,
                            episode_id=episode_idx,
                        )
                    self.system.receive_action(np.hstack((new_action,np.array([1 / (simulator.step_size * simulator.N_steps)]))))
                    toe1.append(running_cost)
                    twap1.append(observation[2,1])
                self.simulator.reset()
                toe.append(toe1)
                twap.append(twap1)
                self.total_costs_episodic.append(self.total_cost)
                self.total_cost = 0
            self.toe = toe
            self.twap = twap
            self.learning_curve.append(np.mean(self.total_costs_episodic))
            self.last_observations = pd.DataFrame(
                index=np.array(self.data_buffer.data["episode_id"]),
                data=np.array(self.data_buffer.data["observation"]),
            )
            self.last_actions = pd.DataFrame(
                index=np.array(self.data_buffer.data["episode_id"]),
                data=np.array(self.data_buffer.data["action"]),
            )
            if self.train_mode:
                self.critic.fit(self.data_buffer)
                self.policy.REINFORCE_step(self.data_buffer)

            means_total_costs.append(np.mean(self.total_costs_episodic))
            change = (means_total_costs[-1] / means_total_costs[-2] - 1) * 100
            sign = "-" if np.sign(change) == -1 else "+"
            print(
                f"Iteration: {iteration_idx + 1} / {self.N_iterations}, "
                + f"mean total cost {round(means_total_costs[-1], 2)}, "
                + f"% change: {sign}{abs(round(change,2))}, "
                + f"last observation: {self.last_observations.groupby(self.last_observations.index).last().mean().values}, "
                + f"TWAP cash: {pd.DataFrame(self.twap).mean().iloc[-1]:.3f}"
                ,
                end="\n",
            )

            self.total_costs_episodic = []

    def plot_data(self):
        """Plot learning results"""

        data = pd.Series(
            index=range(1, len(self.learning_curve) + 1), data=self.learning_curve
        )
        na_mask = data.isna()
        not_na_mask = ~na_mask
        interpolated_values = data.interpolate()
        interpolated_values[not_na_mask] = None
        data.plot(marker="o", markersize=3)
        interpolated_values.plot(linestyle="--")

        plt.title("Total cost by iteration")
        plt.xlabel("Iteration number")
        plt.ylabel("Total cost")
        plt.show()
        
        data = pd.DataFrame(self.toe).mean()
        na_mask = data.isna()
        not_na_mask = ~na_mask
        interpolated_values = data.interpolate()
        interpolated_values[not_na_mask] = None
        data.plot(marker=".", markersize=1)
        interpolated_values.plot(linestyle="--")

        plt.title("Running cost by step in last iteration")
        plt.xlabel("Step Number")
        plt.ylabel("Running cost")
        plt.show()
        N_steps = len(self.last_actions)//self.N_episodes
        self.last_observations.index = np.arange(N_steps).tolist()*self.N_episodes
        theta_ax, dot_theta_ax, cash_ax, time_ax = pd.DataFrame(
            data = self.last_observations.groupby(self.last_observations.index).mean().values,
        columns = ['position', 'price', 'cash RL', 'time']).plot(
            xlabel="Step Number",
            title="Mean observations in last iteration",
            subplots=True,
            grid=True,
        )
        theta_ax.set_ylabel("position")
        dot_theta_ax.set_ylabel("price")
        cash_ax.set_ylabel("cash")
        time_ax.set_ylabel("time")
        
        pd.DataFrame(data = np.array(self.twap).mean(axis = 0), columns = ['cash TWAP']).plot(ax=cash_ax)
        
        self.last_actions.index = np.arange(N_steps).tolist() * self.N_episodes
        actions_ax = pd.DataFrame(
            data = self.last_actions.groupby(self.last_actions.index).mean().values
        ).plot(
            xlabel="Step Number",
            title="Mean actions in last iteration",
            legend=False,
            grid=True,
        )
        actions_ax.set_ylabel("action")
        
        pd.DataFrame(
            self.last_actions.groupby(self.last_actions.index).mean().values
        ).rolling(50).mean().plot(
            ax=actions_ax, legend = False
        )

        plt.show()
        

if __name__=='__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_config', type=str,
                        help='Path to config for the scenario')
    args = parser.parse_args()
    with open(args.path_to_config,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    SEED = 0xC0FFEE
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    system = MarketSystem(**config['MarketSystem'])
    simulator = EM_Simulator(
        system, state_init=np.array([[1.0, 1.0, 0.0, 1.502],[1.0, 1.0, 0.0, 1.502]]).T,
        **config['EM_Simulator']
    )
    discount_factor = config['discount_factor']
    config['GaussianPDFModelCus']['action_bounds'] = np.array([[config['GaussianPDFModelCus']['action_bounds_l'],
                                                                config['GaussianPDFModelCus']['action_bounds_u']]])
    del config['GaussianPDFModelCus']['action_bounds_l']
    del config['GaussianPDFModelCus']['action_bounds_u']

    model = GaussianPDFModelCus(
        dim_observation=system.dim_observation,
        dim_action=system.dim_action,
        **config['GaussianPDFModelCus']
    )
    critic_model = ModelPerceptron(
        dim_input=system.dim_observation,
        **config['ModelPerceptron']
    )
    critic_optimizer = OptimizerSampled(
        model=critic_model,
        opt_method=torch.optim.Adam,
        is_reinstantiate_optimizer=True,
        **config['OptimizerSampledCritic']
    )
    critic = Critic(
        td_n=config['Critic']['td_n'],
        discount_factor=discount_factor,
        device="cpu",
        model=critic_model,
        optimizer=critic_optimizer,
    )
    policy_optimizer = OptimizerSampled(
        model=model,
        opt_method=torch.optim.Adam,
        is_reinstantiate_optimizer=False,
        **config['OptimizerSampledPolicy']
    )
    policy = Policy(
        model,
        policy_optimizer,
        critic=critic,
        discount_factor=discount_factor,
    )
    # This termination criterion never terminates episodes
    trivial_terminantion_criterion = lambda *args: False

    scenario = MonteCarloSimulationScenario(
        train_mode=True,
        simulator=simulator,
        system=system,
        policy=policy,
        critic=critic,
        termination_criterion=trivial_terminantion_criterion,
        discount_factor=discount_factor,
        **config['MonteCarloSimulationScenarioTrain']
    )
    print('Running the model:')
    try:
        scenario.run()
    except KeyboardInterrupt:
        scenario.plot_data()

    scenario.plot_data()
    
    print('Train Final average CASH:', scenario.last_observations.groupby(scenario.last_observations.index).mean()[2].values[-1])
    
    print('Train TWAP final mean cash:', pd.DataFrame(scenario.twap).mean().iloc[-1])
    
    scenario2 = MonteCarloSimulationScenario(
        train_mode=False,
        simulator=simulator,
        system=system,
        policy=policy,
        critic=critic,
        termination_criterion=trivial_terminantion_criterion,
        discount_factor=discount_factor,
        **config['MonteCarloSimulationScenarioTest']
    )

    try:
        scenario2.run()
    except KeyboardInterrupt:
        scenario2.plot_data()

    scenario2.plot_data()
    
    print('Test Final average CASH:', scenario2.last_observations.groupby(scenario2.last_observations.index).mean()[2].values[-1])
    print('Test TWAP Final average CASH:', pd.DataFrame(scenario2.twap).mean().iloc[-1])