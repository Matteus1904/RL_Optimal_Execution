import numpy as np

class MarketSystem:
    """System class: Simulated market. State transition function"""

    dim_action: int = 1
    dim_observation: int = 4
    dim_state: int = 4

    b: float = 0.01
    sigma: float = 0.3
    k: float = 0.1

    def __init__(self, tpi, ppi, volatility, tc, inventory_risk) -> None:
        """Initialize `MarketSystem`"""
        self.b = ppi
        self.k = tpi
        self.sigma = volatility
        self.tc = tc
        self.rho = inventory_risk
        self.reset()

    def reset(self) -> None:
        """Reset system to inital state."""

        self.action = np.zeros(self.dim_action)

    def compute_dynamics(self, state: np.array, action: np.array,  step_size, term: bool) -> np.array:
        """Calculate right-hand-side for Euler integrator

        Args:
            state (np.array): current state
            action (np.array): current action

        Returns:
            np.array: right-hand-side for Euler integrator
        """

        Dstate = np.zeros((self.dim_observation,2))

        if term:
            action = state[0] / step_size 

        Dstate[0] = -action * step_size
        Dstate[1] = -self.b * action * step_size + self.sigma * np.random.normal(0,1) * np.sqrt(step_size) 
        Dstate[2] = ((state[1] + Dstate[1]) - self.k * np.abs(action)**0.5 ) * action * step_size - np.abs( action * step_size) * self.tc - self.rho*(state[0]+Dstate[0])**2
        Dstate[3] = -step_size
        return Dstate
    

    def compute_closed_loop_rhs(self, state: np.array, step_size: np.float32, term: bool) -> np.array:
        """Get right-hand-side for current observation and saved `self.action`

        Args:
            state (np.array): current state

        Returns:
            np.array: right-hand-side for Euler integrator
        """

        system_right_hand_side = self.compute_dynamics(state, self.action,  step_size, term)
        return system_right_hand_side
    

    def receive_action(self, action: np.array) -> None:
        """Save current action to `self.action`

        Args:
            action (np.array): current action
        """

        self.action = action

    @staticmethod
    def get_observation(state : np.array) -> np.array:
        """Get observation given a state

        Args:
            state (np.array): system state

        Returns:
            np.array: observation
        """
        observation = state

        return observation
    
    

class HistoricalMarketSystem:
    """System class: historical market. State transition function"""

    dim_action: int = 1
    dim_observation: int = 5
    dim_state: int = 5

    b: float = 0.01
    sigma: float = 0.3
    k: float = 0.1

    def __init__(self, data, tpi, ppi, volatility, tc) -> None:
        """Initialize `MarketSystem`"""
        self.b = ppi
        self.k = tpi
        self.sigma = volatility
        self.tc = tc
        self.data = data
        self.reset()
        self.executed_price = 25000
        self.last_executed_price = 25000
        self.episode_idx = 0
        self.action = np.zeros(2)

    def executed_price_calc(self, step_idx, action):
        bids_qtys = np.array([
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_1_qty"],
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_2_qty"],
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_3_qty"],
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_4_qty"],
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_5_qty"],
        ]).cumsum()
        bids_prices = np.array([
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_1_px"],
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_2_px"],
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_3_px"],
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_4_px"],
            self.data.iloc[step_idx + 1000 * self.episode_idx]["bid_5_px"],
        ])
        executed_at_level = np.argmax(bids_qtys - action > -1e-4)
        if action > bids_qtys[-1]:
            executed_at_level = -1
        self.executed_price = bids_prices[executed_at_level]
        

    def reset(self) -> None:
        """Reset system to inital state."""

        self.action = np.zeros(self.dim_action)

    def compute_dynamics(self, state: np.array, action: np.array,  step_size, step_idx: int, term: bool) -> np.array:
        self.last_executed_price = self.executed_price
        self.executed_price_calc(step_idx, action[0])
        Dstate = np.zeros((self.dim_observation,2))
        
        if term:
            action = state[0] / step_size 

        Dstate[0] = -action * step_size
        ### executed price
        if step_idx == 0:
            Dstate[1] = 0
        else:
            Dstate[1] = self.data["mid_price"].iloc[step_idx + 1000 * self.episode_idx] - self.data["mid_price"].iloc[step_idx + 1000 * self.episode_idx-1]
        ### cash
        Dstate[2] = ((state[1] + Dstate[1]) - self.k * np.abs(action)**0.5 ) * action * step_size - np.abs( action * step_size) * self.tc
        
        Dstate[3] = self.executed_price - self.last_executed_price
        Dstate[4] = ((state[1] + Dstate[3]) - self.k * np.abs(action)**0.5 ) * action * step_size - np.abs( action * step_size) * self.tc

        return Dstate
    

    def compute_closed_loop_rhs(self, state: np.array, step_size: np.float32, step_idx: int, term: bool) -> np.array:
        """Get right-hand-side for current observation and saved `self.action`

        Args:
            state (np.array): current state

        Returns:
            np.array: right-hand-side for Euler integrator
        """
        system_right_hand_side = self.compute_dynamics(state, self.action,  step_size, step_idx, term)
        return system_right_hand_side

    def receive_action(self, action: np.array) -> None:
        """Save current action to `self.action`

        Args:
            action (np.array): current action
        """

        self.action = action


    @staticmethod
    def get_observation(state : np.array) -> np.array:
        """Get observation given a state

        Args:
            state (np.array): system state

        Returns:
            np.array: observation
        """
        observation = state
        return observation