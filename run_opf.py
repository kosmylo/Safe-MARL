from utils.create_net import create_network
from utils.opf import opf_model
from utils.plot_res import plot_optimization_results
import yaml
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# load env args
with open("./madrl/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]

# Load the network data
network_data = create_network()

# Define the additional inputs for the OPF model
flex_price = {t: 0.20 if 17 <= t <= 21 else 0.10 for t in range(1, env_config_dict['episode_limit'] + 1)}
load_profile = [0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
pv_profile = [0, 0, 0, 0.1, 0.3, 0.5, 0.8, 0.9, 1.0, 0.9, 0.8, 0.6, 0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Prepare the active and reactive power demands for each node
active_power_demand = {n: [network_data['active_power_demand'][n] * load_profile[t] for t in range(env_config_dict['episode_limit'])] for n in network_data['bus_numbers']}
reactive_power_demand = {n: [network_data['reactive_power_demand'][n] * load_profile[t] for t in range(env_config_dict['episode_limit'])] for n in network_data['bus_numbers']}
pv_active_power = {g: [env_config_dict['pv_cap'] * pv_profile[t] for t in range(env_config_dict['episode_limit'])] for g in network_data['PVs_at_buildings']}

# Set the initial energy of ESS (for simplicity, starting at E_min)
initial_ess_energy = {k: env_config_dict['e_min'] for k in network_data['ESSs_at_buildings']}

results = opf_model(network_data, flex_price, active_power_demand, reactive_power_demand, pv_active_power, initial_ess_energy)

logger.info('Optimization results: %s', results)

plot_optimization_results(results)