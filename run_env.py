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

    # Initialize lists to store real action values for each timestep
    all_percentage_reduction = []
    all_ess_charging = []
    all_ess_discharging = []
    all_q_pv = []

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

        actions = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.normal(0, 0.5, n_actions)
            action = action[avail_actions_ind]
            actions.append(action)

        actions = np.concatenate(actions, axis=0)

        reward, _, info  = env.step(actions)

        episode_reward += reward

        rewards.append(episode_reward)

        # Accessing the values directly from the environment's state
        percentage_reduction = env.percentage_reduction
        ess_charging = env.ess_charging
        ess_discharging = env.ess_discharging
        q_pv = env.q_pv

        # Append the processed action values to the respective lists
        all_percentage_reduction.append(percentage_reduction)
        all_ess_charging.append(ess_charging)
        all_ess_discharging.append(ess_discharging)
        all_q_pv.append(q_pv)

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

        # Log the real values inside the logger.info() calls
        logger.info(f"Real Active Demand at timestep {t} (kW): "
                    f"{ {bus: active_demand[bus] * env_config_dict['s_nom'] for bus in active_demand} }")
        logger.info(f"Real Reactive Demand at timestep {t} (kVar): "
                    f"{ {bus: reactive_demand[bus] * env_config_dict['s_nom'] for bus in reactive_demand} }")
        logger.info(f"Real PV Power at timestep {t} (kW): "
                    f"{ {pv: pv_power[pv] * env_config_dict['s_nom'] for pv in pv_power} }")
        logger.info(f"Voltages at timestep {t} (pu): {voltages}")
        logger.info(f"Prices at timestep {t} (Euros/kWh): {prices}")
        logger.info(f"ESS Energy at timestep {t} (kWh): "
                    f"{ {ess: ess_energy[ess] * env_config_dict['s_nom'] for ess in ess_energy} }")
        logger.info(f"Power Reduction at timestep {t} (kW): "
                    f"{ {building: percentage_reduction[building] * active_demand[building] * env_config_dict['s_nom'] for building in percentage_reduction} }")
        logger.info(f"ESS Charging at timestep {t} (kW): "
                    f"{ {ess: ess_charging[ess] * env_config_dict['s_nom'] for ess in ess_charging} }")
        logger.info(f"ESS Discharging at timestep {t} (kW): "
                    f"{ {ess: ess_discharging[ess] * env_config_dict['s_nom'] for ess in ess_discharging} }")
        logger.info(f"Reactive Power from PV at timestep {t} (kVar): "
                    f"{ {pv: q_pv[pv] * env_config_dict['s_nom'] for pv in q_pv} }")
        logger.info(f"Reward at timestep {t}: {reward}")
    
    logger.info(f"Total reward in episode {e} = {episode_reward:.2f}")
    
    # Plot results
    plot_environment_results(all_active_demand, all_reactive_demand, all_pv_power, 
                             all_voltages, all_prices, all_ess_energy, rewards, all_percentage_reduction, 
                             all_ess_charging, all_ess_discharging, all_q_pv
                             )

env.close()