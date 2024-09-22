from ..multiagentenv import MultiAgentEnv
import numpy as np
from utils.create_net import create_network
from utils.pf import power_flow_solver
import pandas as pd
import os
from math import acos, tan
import copy
from collections import namedtuple
import logging

run_env_logger = logging.getLogger('RunEnvLogger')

def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

class ActionSpace(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

class FlexibilityProvisionEnv(MultiAgentEnv):
    """this class is for the environment of distributed flexibility provision

        it is easy to interact with the environment, e.g.,

        state, global_state = env.reset()
        for t in range(240):
            actions = agents.get_actions(state) # a vector involving all agents' actions
            reward, done, info = env.step(actions)
            next_state = env.get_obs()
            state = next_state
    """
    def __init__(self, kwargs):
        """initialisation"""
        # unpack args
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
    
        # Set model_type based on the 'alg' argument
        self.model = getattr(self.args, 'alg', None) 

        # set the data path
        self.data_path = args.data_path

        # set the random seed
        np.random.seed(args.seed)
        
        # load the model of power network
        self.base_powergrid = self._load_network()

        # load data
        self.pv_data = self._load_pv_data()
        self.active_demand_data = self._load_active_demand_data()
        self.reactive_demand_data = self._load_reactive_demand_data()
        self.price_data = self._load_price_data()

        # define episode
        self.episode_limit = args.episode_limit
    
        # define action space and observation space
        self.action_space = ActionSpace(low=self.args.action_low, high=self.args.action_high)
        self.history = args.history
        self.n_agents = len(self.base_powergrid['buildings'])
        self.n_actions = 4  # P_red, P_esc, P_esd, Q_pv for each building
        self.agent_ids = self.base_powergrid['buildings']  # Agent IDs equal to bus IDs with buildings
        agents_obs, state = self.reset()

        self.obs_size = agents_obs[0].shape[0]
        self.state_size = state.shape[0]

    def reset(self):
        """Reset the environment."""
        self.steps = 1
        self.cumulative_reward = 0

        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}
        
        solvable = False
        while not solvable:
            # Reset the time stamp 
            self.start_hour = self._select_start_hour()
            self.start_day = self._select_start_day()
            self.start_interval = self._select_start_interval()

            run_env_logger.info(f"Episode starts at day {self.start_day}, hour {self.start_hour}, interval {self.start_interval}")
        
            # Get one episode of data
            self.pv_history = self._get_episode_pv_history()
            self.active_demand_history = self._get_episode_active_demand_history()
            self.reactive_demand_history = self._get_episode_reactive_demand_history()
            self.price_history = self._get_episode_price_history()
            
            # Set demand, PV, and prices for the first time step
            self._set_demand_pv_prices()

            self.initial_ess_energy = {k: np.random.uniform(0.9 * (self.args.e_max / 2), 1.1 * (self.args.e_max / 2)) for k in self.base_powergrid['ESSs_at_buildings']}

            # Use get_action to randomize initial actions
            actions = self.get_action()

            num_buildings = len(self.base_powergrid['buildings'])

            # Initialize dictionaries to hold the parsed actions
            self.percentage_reduction = {}
            self.ess_charging = {}
            self.ess_discharging = {}
            self.q_pv = {}

            for i in range(num_buildings):
                # Each agent controls 4 actions
                self.percentage_reduction[self.base_powergrid['buildings'][i]] = self.args.max_power_reduction * actions[i * 4]
                self.ess_charging[self.base_powergrid['ESSs_at_buildings'][i]] = self.args.p_ch_max * actions[i * 4 + 1]
                self.ess_discharging[self.base_powergrid['ESSs_at_buildings'][i]] = self.args.p_dis_max * actions[i * 4 + 2]
                self.q_pv[self.base_powergrid['PVs_at_buildings'][i]] = self._scale_and_clip_q_pv(actions[i * 4 + 3], self.current_pv_power[self.base_powergrid['PVs_at_buildings'][i]])

            # Clip the power reduction percentages to be within the allowed range
            self.percentage_reduction = self.clip_percentage_reduction(self.percentage_reduction)
            
            # Compute power reduction based on active power demand and percentage reduction
            self.power_reduction = {building: self.current_active_demand[building] * self.percentage_reduction[building] for building in self.base_powergrid['buildings']}

            # Adjust ESS actions to prevent simultaneous charging and discharging
            self.ess_charging, self.ess_discharging = self.adjust_ess_actions(self.ess_charging, self.ess_discharging)

            for k in self.ess_charging:
                self.ess_charging[k], self.ess_discharging[k] = self._clip_power_charging_discharging(self.ess_charging[k], self.ess_discharging[k], self.initial_ess_energy[k])
            
            try:
                # Attempt to solve the initial power flow with the randomized actions
                result = power_flow_solver(
                    self.base_powergrid,
                    self.current_active_demand,
                    self.current_reactive_demand,
                    self.power_reduction,
                    self.current_pv_power,
                    self.q_pv,
                    self.ess_charging,
                    self.ess_discharging,
                    self.initial_ess_energy
                )
                
                self.current_voltage = result['Voltages']
                self.current_ess_energy = result['Next ESS Energy']

                solvable = True
            except Exception as e:
                print("The power flow for the current initialization cannot be solved.")
                print(e)
                solvable = False
        
        return self.get_obs(), self.get_state()
    
    def manual_reset(self, day, hour, interval):
        """Manually reset the environment with a specified start time."""
        # Reset the time step, cumulative rewards, and observation history
        self.steps = 1
        self.cumulative_reward = 0
        
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # Manually set the time stamp
        self.start_day = day
        self.start_hour = hour
        self.start_interval = interval

        run_env_logger.info(f"Episode starts at day {self.start_day}, hour {self.start_hour}, interval {self.start_interval}")

        # Get one episode of data
        self.pv_history = self._get_episode_pv_history()
        self.active_demand_history = self._get_episode_active_demand_history()
        self.reactive_demand_history = self._get_episode_reactive_demand_history()
        self.price_history = self._get_episode_price_history()

        # Set demand, PV, and prices for the first time step
        self._set_demand_pv_prices()

        self.initial_ess_energy = {k: np.random.uniform(0.9 * (self.args.e_max / 2), 1.1 * (self.args.e_max / 2)) for k in self.base_powergrid['ESSs_at_buildings']}

        # Use get_action to randomize initial actions
        actions = self.get_action()

        num_buildings = len(self.base_powergrid['buildings'])

        # Initialize dictionaries to hold the parsed actions
        self.percentage_reduction = {}
        self.ess_charging = {}
        self.ess_discharging = {}
        self.q_pv = {}

        for i in range(num_buildings):
            # Each agent controls 4 actions
            self.percentage_reduction[self.base_powergrid['buildings'][i]] = self.args.max_power_reduction * actions[i * 4]
            self.ess_charging[self.base_powergrid['ESSs_at_buildings'][i]] = self.args.p_ch_max * actions[i * 4 + 1]
            self.ess_discharging[self.base_powergrid['ESSs_at_buildings'][i]] = self.args.p_dis_max * actions[i * 4 + 2]
            self.q_pv[self.base_powergrid['PVs_at_buildings'][i]] = self._scale_and_clip_q_pv(actions[i * 4 + 3], self.current_pv_power[self.base_powergrid['PVs_at_buildings'][i]])

        # Clip the power reduction percentages to be within the allowed range
        self.percentage_reduction = self.clip_percentage_reduction(self.percentage_reduction)

        # Compute power reduction based on active power demand and percentage reduction
        self.power_reduction = {building: self.current_active_demand[building] * self.percentage_reduction[building] for building in self.base_powergrid['buildings']}

        # Adjust ESS actions to prevent simultaneous charging and discharging
        self.ess_charging, self.ess_discharging = self.adjust_ess_actions(self.ess_charging, self.ess_discharging)

        for k in self.ess_charging:
            self.ess_charging[k], self.ess_discharging[k] = self._clip_power_charging_discharging(self.ess_charging[k], self.ess_discharging[k], self.initial_ess_energy[k])

        solvable = False
        while not solvable:
            try:
                # Attempt to solve the initial power flow with the randomized actions
                result = power_flow_solver(
                    self.base_powergrid,
                    self.current_active_demand,
                    self.current_reactive_demand,
                    self.power_reduction,
                    self.current_pv_power,
                    self.q_pv,
                    self.ess_charging,
                    self.ess_discharging,
                    self.initial_ess_energy
                )
                
                self.current_voltage = result['Voltages']
                self.current_ess_energy = result['Next ESS Energy']

                solvable = True
            except Exception as e:
                print("The power flow for the current initialization cannot be solved.")
                print(e)
                solvable = False

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """Take a step in the environment with the given actions."""
        num_buildings = len(self.base_powergrid['buildings'])

        # Save all relevant state variables to roll back in case of solver failure
        last_valid_state = {
            'voltages': copy.deepcopy(self.current_voltage),
            'ess_energy': copy.deepcopy(self.current_ess_energy),
            'active_demand': copy.deepcopy(self.current_active_demand),
            'reactive_demand': copy.deepcopy(self.current_reactive_demand),
            'pv_power': copy.deepcopy(self.current_pv_power),
            'price': copy.deepcopy(self.current_price),
            'power_reduction': copy.deepcopy(self.power_reduction),
            'ess_charging': copy.deepcopy(self.ess_charging),
            'ess_discharging': copy.deepcopy(self.ess_discharging),
            'q_pv': copy.deepcopy(self.q_pv)
        }

        # Reshape actions to match the expected shape
        actions = actions.reshape((self.n_agents*self.n_actions))

        # Initialize dictionaries to hold the parsed actions
        self.percentage_reduction = {}
        self.ess_charging = {}
        self.ess_discharging = {}
        self.q_pv = {}

        if self.model == 'safemaddpg':
            # Directly use actions for safemaddpg
            for i in range(num_buildings):
                self.percentage_reduction[self.base_powergrid['buildings'][i]] = actions[i * 4]
                self.ess_charging[self.base_powergrid['ESSs_at_buildings'][i]] = actions[i * 4 + 1]
                self.ess_discharging[self.base_powergrid['ESSs_at_buildings'][i]] = actions[i * 4 + 2]
                self.q_pv[self.base_powergrid['PVs_at_buildings'][i]] = actions[i * 4 + 3]
        else:
            for i in range(num_buildings):
                # Scale the actions for the rest of the models
                self.percentage_reduction[self.base_powergrid['buildings'][i]] = self.args.max_power_reduction * actions[i * 4]
                self.ess_charging[self.base_powergrid['ESSs_at_buildings'][i]] = self.args.p_ch_max * actions[i * 4 + 1]
                self.ess_discharging[self.base_powergrid['ESSs_at_buildings'][i]] = self.args.p_dis_max * actions[i * 4 + 2]
                self.q_pv[self.base_powergrid['PVs_at_buildings'][i]] = self._scale_and_clip_q_pv(actions[i * 4 + 3], self.current_pv_power[self.base_powergrid['PVs_at_buildings'][i]])

        # Clip the power reduction percentages to be within the allowed range
        self.percentage_reduction = self.clip_percentage_reduction(self.percentage_reduction)
            
        # Adjust ESS actions to prevent simultaneous charging and discharging
        self.ess_charging, self.ess_discharging = self.adjust_ess_actions(self.ess_charging, self.ess_discharging)

        for k in self.ess_charging:
            self.ess_charging[k], self.ess_discharging[k] = self._clip_power_charging_discharging(self.ess_charging[k], self.ess_discharging[k], self.current_ess_energy[k])
        
        # Compute power reduction based on active power demand and percentage reduction
        self.power_reduction = {building: self.current_active_demand[building] * self.percentage_reduction[building] for building in self.base_powergrid['buildings']}

        # Try to solve the power flow with the applied actions
        solvable = False
        try:
            result = power_flow_solver(
                self.base_powergrid,
                self.current_active_demand,
                self.current_reactive_demand,
                self.power_reduction,
                self.current_pv_power,
                self.q_pv,
                self.ess_charging,
                self.ess_discharging,
                self.initial_ess_energy
            )

            self.current_voltage = result['Voltages']
            self.current_ess_energy = result['Next ESS Energy']

            solvable = True
        except Exception as e:
            print("The power flow for the current step cannot be solved.")
            print(e)

            # Restore the state to the last valid state
            self.current_voltage = last_valid_state['voltages']
            self.current_ess_energy = last_valid_state['ess_energy']
            self.current_active_demand = last_valid_state['active_demand']
            self.current_reactive_demand = last_valid_state['reactive_demand']
            self.current_pv_power = last_valid_state['pv_power']
            self.current_price = last_valid_state['price']
            self.power_reduction = last_valid_state['power_reduction']
            self.ess_charging = last_valid_state['ess_charging']
            self.ess_discharging = last_valid_state['ess_discharging']
            self.q_pv = last_valid_state['q_pv']
        
        if solvable:
        # Calculate reward and gather info
            reward, info = self.calculate_reward(self.power_reduction, self.ess_charging, self.ess_discharging, self.q_pv, self.current_voltage)
        else:
            # Penalize if unsolvable
            reward, info = self.calculate_reward(self.power_reduction, self.ess_charging, self.ess_discharging, self.q_pv, self.current_voltage)
            reward -= 200
            info["solver_failed"] = True
        
        # Update demand, PV, and prices for the next time step
        self._set_demand_pv_prices()

        self.steps += 1
        self.cumulative_reward += reward

        if self.steps >= self.episode_limit or not solvable:
            terminated = True
        else:
            terminated = False
        
        if terminated:
            print (f"Episode terminated at time: {self.steps} with return: {self.cumulative_reward:2.4f}.")

        # Update initial_ess_energy for the next step
        self.initial_ess_energy = self.current_ess_energy

        return reward, terminated, info

    def get_state(self):
        """ Returns the global state of the environment. """
        state = []
        state += [self.current_active_demand[bus] for bus in self.base_powergrid['bus_numbers']]
        state += [self.current_reactive_demand[bus] for bus in self.base_powergrid['bus_numbers']]
        state += [self.current_pv_power[pv] for pv in self.base_powergrid['PVs_at_buildings']]
        state += [self.current_voltage[bus] for bus in self.base_powergrid['bus_numbers']]
        state += [self.current_price[0]] 
        state += [self.current_ess_energy[ess] for ess in self.base_powergrid['ESSs_at_buildings']]
        state = np.array(state)
        return state
    
    def get_obs(self):
        """Returns the list of observations for all agents."""
        agents_obs = []
        
        for i in range(self.n_agents):
            obs = []
            building = self.agent_ids[i]
            obs.append(self.current_active_demand[building])
            obs.append(self.current_reactive_demand[building])
            obs.append(self.current_pv_power[building])
            obs.append(self.current_voltage[building])
            obs.append(self.current_price[0])
            obs.append(self.current_ess_energy[building])
            
            agents_obs.append(np.array(obs))

        # Handle historical observations if history > 1
        if self.history > 1:
            agents_obs_history = []
            for i, obs in enumerate(agents_obs):
                if len(self.obs_history[i]) >= self.history - 1:
                    # Combine the last history - 1 observations with the current one
                    obs_hist = np.concatenate(self.obs_history[i][-self.history + 1:] + [obs], axis=0)
                else:
                    # Pad with zeros if there aren't enough historical observations
                    zeros = [np.zeros_like(obs)] * (self.history - len(self.obs_history[i]) - 1)
                    obs_hist = np.concatenate(zeros + self.obs_history[i] + [obs], axis=0)
                
                agents_obs_history.append(copy.deepcopy(obs_hist))
                self.obs_history[i].append(copy.deepcopy(obs))
        
            agents_obs = agents_obs_history
        
        return agents_obs

    def get_obs_agent(self, agent_id):
        """Returns the observation for a specific agent."""
        agents_obs = self.get_obs()
        return agents_obs[agent_id]
    
    def _select_start_hour(self):
        """select start hour for an episode"""
        return np.random.choice(24)
    
    def _select_start_interval(self):
        """select start interval for an episode"""
        return np.random.choice( 60 // self.time_delta )

    def _select_start_day(self):
        """select start day (date) for an episode"""
        pv_data = self.pv_data
        pv_days = (pv_data.index[-1] - pv_data.index[0]).days
        self.time_delta = (pv_data.index[1] - pv_data.index[0]).seconds // 60
        episode_days = ( self.episode_limit // (24 * (60 // self.time_delta) ) ) + 1  # margin
        return np.random.choice(pv_days - episode_days)

    def _load_network(self):
        """load network"""
        network_data = create_network()
        return network_data

    def _load_pv_data(self):
        """load pv data"""
        pv_path = os.path.join(self.data_path, 'pv_active.csv')
        pv = pd.read_csv(pv_path, index_col=None)
        pv.index = pd.to_datetime(pv.iloc[:, 0])
        pv.index.name = 'time'
        pv = pv.iloc[::1, 1:] * self.args.pv_scale
        return self.resample_data(pv, self.args.sample_interval)

    def _load_active_demand_data(self):
        """load active demand data"""
        demand_path = os.path.join(self.data_path, 'load_active.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return self.resample_data(demand, self.args.sample_interval)
    
    def _load_reactive_demand_data(self):
        """load reactive demand data"""
        demand_path = os.path.join(self.data_path, 'load_reactive.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.reactive_scale
        return self.resample_data(demand, self.args.sample_interval)
    
    def _load_price_data(self):
        """load price data"""
        price_path = os.path.join(self.data_path, 'prices.csv')
        price = pd.read_csv(price_path, index_col=None)
        price.index = pd.to_datetime(price.iloc[:, 0])
        price.index.name = 'time'
        price = price.iloc[::1, 1:]
        return self.resample_data(price, self.args.sample_interval)
    
    def resample_data(self, data, interval):
        """Resample data to the specified interval."""
        data = data.resample(interval).mean()
        data = data.interpolate(method='linear')
        return data
    
    def _get_episode_active_demand_history(self):
        """Return the active power histories for all loads in an episode."""
        episode_length = self.episode_limit
        history = self.history
        start = self.start_interval + self.start_hour * (60 // self.time_delta) + self.start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1

        # Extract the active demand data for the episode
        active_demand_history = self.active_demand_data[start:start + nr_intervals]

        # Log the timestamps and shape for the episode
        timestamps = active_demand_history.index
        run_env_logger.info(f"Episode Active Demand history timestamps: {timestamps}")
        run_env_logger.info(f"Active Demand history shape: {active_demand_history.shape}")

        active_demand_history = active_demand_history.values
        slack_bus_demand = np.zeros((active_demand_history.shape[0], 1))  # Adding slack bus with zero demand
        active_demand_history = np.hstack((slack_bus_demand, active_demand_history))
        
        return active_demand_history.reshape(-1, len(self.base_powergrid['bus_numbers']))

    def _get_episode_reactive_demand_history(self):
        """Return the reactive power histories for all loads in an episode."""
        episode_length = self.episode_limit
        history = self.history
        start = self.start_interval + self.start_hour * (60 // self.time_delta) + self.start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1

        # Extract the reactive demand data for the episode
        reactive_demand_history = self.reactive_demand_data[start:start + nr_intervals]

        # Log the timestamps and shape for the episode
        timestamps = reactive_demand_history.index
        run_env_logger.info(f"Episode Reactive Demand history timestamps: {timestamps}")
        run_env_logger.info(f"Reactive Demand history shape: {reactive_demand_history.shape}")

        reactive_demand_history = reactive_demand_history.values
        slack_bus_demand = np.zeros((reactive_demand_history.shape[0], 1))  # Adding slack bus with zero demand
        reactive_demand_history = np.hstack((slack_bus_demand, reactive_demand_history))
        
        return reactive_demand_history.reshape(-1, len(self.base_powergrid['bus_numbers']))

    def _get_episode_pv_history(self):
        """Return the PV history in an episode."""
        episode_length = self.episode_limit
        history = self.history
        start = self.start_interval + self.start_hour * (60 // self.time_delta) + self.start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1
        
        # Extract the PV data for the episode
        pv_history = self.pv_data[start:start + nr_intervals]

        # Log the timestamps for the episode
        timestamps = pv_history.index
        run_env_logger.info(f"Episode PV history timestamps: {timestamps}")
        run_env_logger.info(f"PV history shape: {pv_history.shape}")
        
        return pv_history.values

    def _get_episode_price_history(self):
        """Return the price history in an episode."""
        episode_length = self.episode_limit
        history = self.history
        start = self.start_interval + self.start_hour * (60 // self.time_delta) + self.start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1

        # Extract the PV data for the episode
        price_history = self.price_data[start:start + nr_intervals]

        # Log the timestamps for the episode
        timestamps =  price_history.index
        run_env_logger.info(f"Episode price history timestamps: {timestamps}")
        run_env_logger.info(f"Price history shape: {price_history.shape}")

        return price_history.values
    
    def _get_active_demand_history(self):
        """Return the history demand for the current step."""
        t = self.steps
        history = self.history

        # Get the current slice of active demand history
        active_demand_history = self.active_demand_history[t:t+history, :]

        # Log the current step, history size, and shape of the returned slice
        run_env_logger.info(f"Current step: {t}, History size: {history}")
        run_env_logger.info(f"Active Demand history shape at step {t}: {active_demand_history.shape}")
        run_env_logger.info(f"Active Demand history at step {t}: {active_demand_history}")

        return active_demand_history

    def _get_reactive_demand_history(self):
        """Return the history reactive demand for the current step."""
        t = self.steps
        history = self.history

        # Get the current slice of reactive demand history
        reactive_demand_history = self.reactive_demand_history[t:t+history, :]

        # Log the current step, history size, and shape of the returned slice
        run_env_logger.info(f"Current step: {t}, History size: {history}")
        run_env_logger.info(f"Reactive Demand history shape at step {t}: {reactive_demand_history.shape}")
        run_env_logger.info(f"Reactive Demand history at step {t}: {reactive_demand_history}")
        
        return reactive_demand_history

    def _get_pv_history(self):
        """Return the PV history for the current step."""
        t = self.steps
        history = self.history

        # Get the current slice of PV history
        pv_history = self.pv_history[t:t+history, :]

        # Log the current step, history size, and shape of the returned slice
        run_env_logger.info(f"Current step: {t}, History size: {history}")
        run_env_logger.info(f"PV history shape at step {t}: {pv_history.shape}")
        run_env_logger.info(f"PV history at step {t}: {pv_history}")

        return pv_history

    def _get_price_history(self):
        """Return the price history for the current step."""
        t = self.steps
        history = self.history

        # Get the current slice of price history
        price_history = self.price_history[t:t+history, :]

        # Log the current step, history size, and shape of the returned slice
        run_env_logger.info(f"Current step: {t}, History size: {history}")
        run_env_logger.info(f"Price history shape at step {t}: {price_history.shape}")
        run_env_logger.info(f"Price history at step {t}: {price_history}")

        return price_history
    
    def _set_demand_pv_prices(self):
        """Update the demand, PV production, and prices according to the histories."""
        pv = copy.copy(self._get_pv_history()[0, :])
        active_demand = copy.copy(self._get_active_demand_history()[0, :])
        reactive_demand = copy.copy(self._get_reactive_demand_history()[0, :])

        # Update the current state in the environment
        self.current_pv_power = {pv_id: pv[i] for i, pv_id in enumerate(self.base_powergrid['PVs_at_buildings'])}
        self.current_active_demand = {bus: active_demand[i] for i, bus in enumerate(self.base_powergrid['bus_numbers'])}
        self.current_reactive_demand = {bus: reactive_demand[i] for i, bus in enumerate(self.base_powergrid['bus_numbers'])}
        self.current_price = self._get_price_history()[0]

    def _scale_and_clip_q_pv(self, reactive_action, active_power):
        """Scale and clip the reactive power based on the action and active power."""
        reactive_power_constraint = tan(acos(self.args.cos_phi_max)) * active_power
        q_pv_min = -reactive_power_constraint
        q_pv_max = reactive_power_constraint
        return np.clip(q_pv_min + reactive_action * (q_pv_max - q_pv_min), q_pv_min, q_pv_max)

    def _clip_power_charging_discharging(self, charging, discharging, current_ess_energy):
        # Clip charging and discharging to their respective maximum values
        charging = np.clip(charging, 0, self.args.p_ch_max)
        discharging = np.clip(discharging, 0, self.args.p_dis_max)

        # Calculate the next ESS energy
        e_next = current_ess_energy + self.args.eta_ch * charging - (1 / self.args.eta_dis) * discharging

        # Ensure ESS energy remains within bounds
        if e_next > self.args.e_max:
            # Reduce charging or increase discharging to not exceed e_max
            excess_energy = e_next - self.args.e_max
            if charging > excess_energy / self.args.eta_ch:
                charging -= excess_energy / self.args.eta_ch
            else:
                discharging += (excess_energy - charging * self.args.eta_ch) * self.args.eta_dis
                charging = 0
            e_next = self.args.e_max

        elif e_next < self.args.e_min:
            # Reduce discharging or increase charging to not go below e_min
            lack_energy = self.args.e_min - e_next
            if discharging > lack_energy * self.args.eta_dis:
                discharging -= lack_energy * self.args.eta_dis
            else:
                charging += (lack_energy - discharging / self.args.eta_dis) / self.args.eta_ch
                discharging = 0
            e_next = self.args.e_min

        # Clip charging and discharging to ensure they do not go negative after adjustments
        charging = np.clip(charging, 0, self.args.p_ch_max)
        discharging = np.clip(discharging, 0, self.args.p_dis_max)

        return charging, discharging
    
    def adjust_ess_actions(self, ess_charging, ess_discharging):
        """Adjust ESS actions to prevent simultaneous charging and discharging."""
        for k in ess_charging:
            if ess_charging[k] > 0 and ess_discharging[k] > 0:
                # If both charging and discharging are happening, we need to correct this
                if ess_charging[k] > ess_discharging[k]:
                    ess_charging[k] -= ess_discharging[k]
                    ess_discharging[k] = 0
                else:
                    ess_discharging[k] -= ess_charging[k]
                    ess_charging[k] = 0
        return ess_charging, ess_discharging
    
    def clip_percentage_reduction(self, percentage_reduction):
        return {k: np.clip(v, 0,  self.args.max_power_reduction) for k, v in percentage_reduction.items()}
    
    def calculate_reward(self, power_reduction, ess_charging, ess_discharging, q_pv, voltages):
        lambda_flex = self.current_price
        revenue = sum(lambda_flex * power_reduction[b] for b in power_reduction)
        der_cost = sum(self.args.pv_cost * q_pv[g] for g in q_pv)
        ess_cost = sum(self.args.ess_cost * (ess_charging[k] + ess_discharging[k]) for k in ess_charging)
        discomfort_penalty = sum(self.args.discomfort_coeff * power_reduction[b] for b in power_reduction)
        voltage_penalty = sum(self.args.voltage_coeff * max(0, v - self.args.v_max, self.args.v_min - v) for v in voltages.values())
        reward = revenue - der_cost - ess_cost - discomfort_penalty - voltage_penalty

        # Ensure that values are all scalars
        revenue = float(revenue)
        reward = float(reward)
        der_cost = float(der_cost)
        ess_cost = float(ess_cost)
        discomfort_penalty = float(discomfort_penalty)
        voltage_penalty = float(voltage_penalty)
        
        info = {
            'reward': reward,
            'revenue': revenue,
            'der_cost': der_cost,
            'ess_cost': ess_cost,
            'discomfort_penalty': discomfort_penalty,
            'voltage_penalty': voltage_penalty,
            'cumulative_reward': self.cumulative_reward
        }

        return reward, info

    def get_obs_size(self):
        """return the observation size"""
        return self.obs_size

    def get_state_size(self):
        """return the state size"""
        return self.state_size
    
    def get_action(self):
        """return the action according to a uniform distribution over [action_lower, action_upper)"""
        rand_action = np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.n_agents * self.n_actions)
        return rand_action

    def get_avail_actions(self):
        """return available actions for all agents"""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """Return the available actions for agent_id."""
        return [1] * self.n_actions 

    def get_total_actions(self):
        """return the total number of actions an agent could ever take"""
        return self.n_actions
    
    def get_num_of_agents(self):
        """return the number of agents"""
        return self.n_agents

    def _get_bus_v(self):
        """Return the current bus voltages."""
        return np.array([self.current_voltage[bus] for bus in self.base_powergrid['bus_numbers']])
    
    def _get_bus_active(self):
        """Return the current active power demand at each bus."""
        return np.array([self.current_active_demand[bus] for bus in self.base_powergrid['bus_numbers']])
    
    def _get_bus_reactive(self):
        """Return the current reactive power demand at each bus."""
        return np.array([self.current_reactive_demand[bus] for bus in self.base_powergrid['bus_numbers']])
    
    def _get_pv_active(self):
        """Return the current active power generated by each PV system."""
        return np.array([self.current_pv_power[pv] for pv in self.base_powergrid['PVs_at_buildings']])
    
    def _get_pv_reactive(self):
        """Return the current reactive power generated by each PV system."""
        return np.array([self.q_pv[pv] for pv in self.base_powergrid['PVs_at_buildings']])
    
    def _get_ess_energy(self):
        """Return the current energy stored in each ESS."""
        return np.array([self.current_ess_energy[ess] for ess in self.base_powergrid['ESSs_at_buildings']])
    
    def _get_power_reduction(self):
        """Return the current active power reduction at each building."""
        return np.array([self.power_reduction[building] for building in self.base_powergrid['buildings']])
    
    def _get_ess_charging(self):
        """Return the current charging power for each ESS."""
        return np.array([self.ess_charging[ess] for ess in self.base_powergrid['ESSs_at_buildings']])
    
    def _get_ess_discharging(self):
        """Return the current discharging power for each ESS."""
        return np.array([self.ess_discharging[ess] for ess in self.base_powergrid['ESSs_at_buildings']])
    
    def _get_price(self):
        """Return the current price at this time step."""
        return np.array([self.current_price])