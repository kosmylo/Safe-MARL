from ..multiagentenv import MultiAgentEnv
import numpy as np
from utils.create_net import create_network
from utils.pf import power_flow_solver
import pandas as pd
import os
from math import acos, tan
from collections import namedtuple

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
        """initialisation
        """
        # unpack args
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # set the data path
        self.data_path = args.data_path

        # set the random seed
        np.random.seed(args.seed)
        
        # load the model of power network
        self.base_powergrid = self._load_network()
        self.cos_phi_max = args.cos_phi_max
        self.max_power_reduction = args.max_power_reduction
        
        # load data
        self.sample_interval = args.sample_interval
        self.pv_data = self._load_pv_data()
        self.active_demand_data = self._load_active_demand_data()
        self.reactive_demand_data = self._load_reactive_demand_data()
        self.price_data = self._load_price_data()

        # define episode
        self.episode_limit = args.episode_limit
        self.history = args.history
    
        self.action_space = ActionSpace(low=self.args.action_low, high=self.args.action_high)
        self.n_agents = len(self.base_powergrid['buildings'])
        self.n_actions = 4  # P_red, P_esc, P_esd, Q_pv for each building
        self.agent_ids = self.base_powergrid['buildings']  # Agent IDs equal to bus IDs with buildings
        agents_obs, state = self.reset()

        self.obs_size = agents_obs[0].shape[0]
        self.state_size = state.shape[0]

    def reset(self):
        self.steps = 0
        self.cumulative_reward = 0
        self.set_initial_state()
        return self.get_obs(), self.get_state()
    
    def set_initial_state(self):
        self.start_hour = self._select_start_hour()
        self.start_day = self._select_start_day()
        self.start_interval = self._select_start_interval()
        self.pv_history = self.get_pv_history()
        self.active_demand_history = self.get_active_demand_history()
        self.reactive_demand_history = self.get_reactive_demand_history()
        self.price_history = self.get_price_history()
        self.set_demand_pv_prices()
        self.initial_ess_energy = {k: self.args.e_max/2 for k in self.base_powergrid['ESSs_at_buildings']}
        self.current_voltage = {bus: 1.0 for bus in self.base_powergrid['bus_numbers']}
        self.current_ess_energy = self.initial_ess_energy

    def step(self, actions):
        num_buildings = len(self.base_powergrid['buildings'])

        # Reshape actions to match the expected shape
        actions = actions.reshape((self.n_agents*self.n_actions))

        # Initialize dictionaries to hold the parsed actions
        percentage_reduction = {}
        ess_charging = {}
        ess_discharging = {}
        q_pv = {}

        for i in range(num_buildings):
            # Each agent controls 4 actions
            percentage_reduction[self.base_powergrid['buildings'][i]] = self.args.max_power_reduction * actions[i * 4]
            ess_charging[self.base_powergrid['ESSs_at_buildings'][i]] = self.args.p_ch_max * actions[i * 4 + 1]
            ess_discharging[self.base_powergrid['ESSs_at_buildings'][i]] = self.args.p_dis_max * actions[i * 4 + 2]
            q_pv[self.base_powergrid['PVs_at_buildings'][i]] = self._scale_and_clip_q_pv(actions[i * 4 + 3], self.current_pv_power[self.base_powergrid['PVs_at_buildings'][i]])

        # Clip the power reduction percentages to be within the allowed range
        percentage_reduction = self.clip_percentage_reduction(percentage_reduction)
        
        # Compute power reduction based on active power demand and percentage reduction
        power_reduction = {building: self.current_active_demand[building] * percentage_reduction[building] for building in self.base_powergrid['buildings']}

        # Adjust ESS actions to prevent simultaneous charging and discharging
        ess_charging, ess_discharging = self.adjust_ess_actions(ess_charging, ess_discharging)

        for k in ess_charging:
            ess_charging[k], ess_discharging[k] = self._clip_power_charging_discharging(ess_charging[k], ess_discharging[k], self.current_ess_energy[k])

        result = power_flow_solver(
            self.base_powergrid,
            self.current_active_demand,
            self.current_reactive_demand,
            power_reduction,
            self.current_pv_power,
            q_pv,
            ess_charging,
            ess_discharging,
            self.initial_ess_energy
        )

        voltages = result['Voltages']
        ess_energy = result['Next ESS Energy']

        self.current_voltage = voltages
        self.current_ess_energy = ess_energy

        reward, info = self.calculate_reward(power_reduction, ess_charging, ess_discharging, q_pv, voltages)
        self.cumulative_reward += reward

        self.set_demand_pv_prices()

        # Update observations and state
        obs = self.get_obs()
        state = self.get_state()

        terminated = self.steps >= self.episode_limit

        self.steps += 1

        # Update initial_ess_energy for the next step
        self.initial_ess_energy = ess_energy

        # Store the values as attributes for later access
        self.percentage_reduction = percentage_reduction
        self.ess_charging = ess_charging
        self.ess_discharging = ess_discharging
        self.q_pv = q_pv

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
        
        return agents_obs

    def get_obs_agent(self, agent_id):
        """Returns the observation for a specific agent."""
        # Find the index of the bus that corresponds to the agent_id
        agent_index = self.agent_ids.index(agent_id)
        return self.get_obs()[agent_index]
    
    def _select_start_hour(self):
        """select start hour for an episode
        """
        return np.random.choice(24)
    
    def _select_start_interval(self):
        """select start interval for an episode
        """
        return np.random.choice( 60 // self.time_delta )

    def _select_start_day(self):
        """select start day (date) for an episode
        """
        pv_data = self.pv_data
        pv_days = (pv_data.index[-1] - pv_data.index[0]).days
        self.time_delta = (pv_data.index[1] - pv_data.index[0]).seconds // 60
        episode_days = ( self.episode_limit // (24 * (60 // self.time_delta) ) ) + 1  # margin
        return np.random.choice(pv_days - episode_days)

    def _load_network(self):
        """load network
        """
        network_data = create_network()
        return network_data

    def _load_pv_data(self):
        """load pv data
        """
        pv_path = os.path.join(self.data_path, 'pv_active.csv')
        pv = pd.read_csv(pv_path, index_col=None)
        pv.index = pd.to_datetime(pv.iloc[:, 0])
        pv.index.name = 'time'
        pv = pv.iloc[::1, 1:] * self.args.pv_scale
        return self.resample_data(pv, self.sample_interval)

    def _load_active_demand_data(self):
        """load active demand data
        """
        demand_path = os.path.join(self.data_path, 'load_active.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        # # Drop the first column (DateTime) as it is now set as the index
        # demand = demand.drop(columns=demand.columns[0])

        # # Apply scaling to the DataFrame columns
        # demand = demand * self.args.demand_scale
        return self.resample_data(demand, self.sample_interval)
    
    def _load_reactive_demand_data(self):
        """load reactive demand data
        """
        demand_path = os.path.join(self.data_path, 'load_reactive.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.reactive_scale
        # # Drop the first column (DateTime) as it is now set as the index
        # demand = demand.drop(columns=demand.columns[0])

        # # Apply scaling to the DataFrame columns
        # demand = demand * self.args.reactive_scale
        return self.resample_data(demand, self.sample_interval)
    
    def _load_price_data(self):
        """load price data
        """
        price_path = os.path.join(self.data_path, 'prices.csv')
        price = pd.read_csv(price_path, index_col=None)
        price.index = pd.to_datetime(price.iloc[:, 0])
        price.index.name = 'time'
        price = price.iloc[::1, 1:]
        return self.resample_data(price, self.sample_interval)
    
    def resample_data(self, data, interval):
        """Resample data to the specified interval."""
        data = data.resample(interval).mean()
        data = data.interpolate(method='linear')
        return data

    def get_active_demand_history(self):
        start = self.start_day * 24 + self.start_hour
        end = start + self.episode_limit + self.history
        active_demand_history = self.active_demand_data.iloc[start:end].values
        slack_bus_demand = np.zeros((active_demand_history.shape[0], 1))  # Adding slack bus with zero demand
        active_demand_history = np.hstack((slack_bus_demand, active_demand_history))
        return active_demand_history.reshape(-1, len(self.base_powergrid['bus_numbers']))

    def get_reactive_demand_history(self):
        start = self.start_day * 24 + self.start_hour
        end = start + self.episode_limit + self.history
        reactive_demand_history = self.reactive_demand_data.iloc[start:end].values
        slack_bus_demand = np.zeros((reactive_demand_history.shape[0], 1))  # Adding slack bus with zero demand
        reactive_demand_history = np.hstack((slack_bus_demand, reactive_demand_history))
        return reactive_demand_history.reshape(-1, len(self.base_powergrid['bus_numbers']))

    def get_pv_history(self):
        start = self.start_day * 24 + self.start_hour
        end = start + self.episode_limit + self.history
        return self.pv_data.iloc[start:end].values

    def get_price_history(self):
        start = self.start_day * 24 + self.start_hour
        end = start + self.episode_limit + self.history
        return self.price_data.iloc[start:end].values

    def set_demand_pv_prices(self):
        self.current_pv_power = {pv: self.pv_history[self.steps][i] for i, pv in enumerate(self.base_powergrid['PVs_at_buildings'])}
        self.current_active_demand = {bus: self.active_demand_history[self.steps][i] for i, bus in enumerate(self.base_powergrid['bus_numbers'])}
        self.current_reactive_demand = {bus: self.reactive_demand_history[self.steps][i] for i, bus in enumerate(self.base_powergrid['bus_numbers'])}
        self.current_price = self.price_history[self.steps]

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
        return {k: np.clip(v, 0,  self.max_power_reduction) for k, v in percentage_reduction.items()}
    
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
        
        info = {
            'revenue': revenue,
            'der_cost': der_cost,
            'ess_cost': ess_cost,
            'discomfort_penalty': discomfort_penalty,
            'voltage_penalty': voltage_penalty,
            'cumulative_reward': self.cumulative_reward
        }
        return reward, info

    def get_obs_size(self):
        """return the observation size
        """
        return self.obs_size

    def get_state_size(self):
        """return the state size
        """
        return self.state_size

    def get_avail_actions(self):
        """return available actions for all agents
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """Return the available actions for agent_id."""
        return [1] * self.n_actions 

    def get_total_actions(self):
        return self.n_actions
    
    def get_num_of_agents(self):
        """return the number of agents"""
        return self.n_agents