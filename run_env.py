from MADRL.environments.flex_provision.flexibility_provision_env import FlexibilityProvisionEnv
from utils.plot_res import plot_environment_results
import numpy as np
import yaml
import os
import logging

# Ensure the logs directory exists
log_dir = "logs/run_env_logs"
os.makedirs(log_dir, exist_ok=True)

# Configure the logger
log_file_path = os.path.join(log_dir, "run_env_log.txt")

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler(log_file_path, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# load env args
with open("./MADRL/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
env_config_dict["data_path"] = "/".join(data_path)

# define envs
env = FlexibilityProvisionEnv(env_config_dict)

n_agents = env.get_num_of_agents()
n_actions = env.get_total_actions()

n_episodes = 1

for e in range(n_episodes):
    state, global_state = env.reset()
    max_steps = 24
    episode_reward = 0

    # Initialize lists to store results for each timestep
    all_active_demand = []
    all_reactive_demand = []
    all_pv_power = []
    all_voltages = []
    all_prices = []
    all_ess_energy = []
    rewards = []

    for t in range(max_steps):
        obs = env.get_obs()
        state = env.get_state()

        # Log the state for inspection
        logger.info(f"State at timestep {t}: {state}")

        actions = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.normal(0, 0.5, n_actions)
            action = action[avail_actions_ind]
            actions.append(action)

        actions = np.concatenate(actions, axis=0)

        logger.info(f"Actions at timestep {t}: {actions}")
            
        reward, _, info = env.step(actions)

        episode_reward += reward

        rewards.append(episode_reward)

        # Extract and store results from the state
        num_buses = len(env.base_powergrid['bus_numbers'])
        num_pvs = len(env.base_powergrid['PVs_at_buildings'])
        num_ess = len(env.base_powergrid['ESSs_at_buildings'])

        active_demand_index = 0
        reactive_demand_index = active_demand_index + num_buses
        pv_power_index = reactive_demand_index + num_buses
        voltage_index = pv_power_index + num_pvs
        price_index = voltage_index + num_buses
        ess_energy_index = price_index + 1

        active_demand = {bus: state[active_demand_index + i] for i, bus in enumerate(env.base_powergrid['bus_numbers'])}
        reactive_demand = {bus: state[reactive_demand_index + i] for i, bus in enumerate(env.base_powergrid['bus_numbers'])}
        pv_power = {pv: state[pv_power_index + i] for i, pv in enumerate(env.base_powergrid['PVs_at_buildings'])}
        voltages = {bus: state[voltage_index + i] for i, bus in enumerate(env.base_powergrid['bus_numbers'])}
        prices = state[price_index]
        ess_energy = {ess: state[ess_energy_index + i] for i, ess in enumerate(env.base_powergrid['ESSs_at_buildings'])}

        all_active_demand.append(active_demand)
        all_reactive_demand.append(reactive_demand)
        all_pv_power.append(pv_power)
        all_voltages.append(voltages)
        all_prices.append(prices)
        all_ess_energy.append(ess_energy)

        # Log the extracted values
        logger.info(f"Active Demand at timestep {t}: {active_demand}")
        logger.info(f"Reactive Demand at timestep {t}: {reactive_demand}")
        logger.info(f"PV Power at timestep {t}: {pv_power}")
        logger.info(f"Voltages at timestep {t}: {voltages}")
        logger.info(f"Prices at timestep {t}: {prices}")
        logger.info(f"ESS Energy at timestep {t}: {ess_energy}")
        logger.info(f"Reward at timestep {t}: {reward}")
    
    logger.info(f"Total reward in episode {e} = {episode_reward:.2f}")

    # Plot results
    plot_environment_results(all_active_demand, all_reactive_demand, all_pv_power, all_voltages, all_prices, all_ess_energy, rewards)

env.close()