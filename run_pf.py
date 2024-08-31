from utils.create_net import create_network
from utils.pf import power_flow_solver
from utils.plot_res import plot_power_flow_results
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

# Example Active and Reactive Power Demand in pu
active_power_demand = {node: 0 if network_data['bus_types'][node] == 1 else 0.1 for node in network_data['bus_numbers']}
reactive_power_demand = {node: 0 if network_data['bus_types'][node] == 1 else 0.005 for node in network_data['bus_numbers']}

# Active Power Reduction capabilities at some buildings (Demand Response) in pu
power_reduction = {node: active_power_demand[node] * env_config_dict['max_power_reduction'] for node in env_config_dict['buildings']}

# Photovoltaic Power Generation at certain buses in pu
pv_power = {node: 0.5 * env_config_dict['pv_cap'] for node in env_config_dict['pv_nodes']}

# Photovoltaic Reactive Power Generation (if applicable) in pu
pv_reactive_power = {node: 0 for node in env_config_dict['pv_nodes']} 

# Energy Storage System Charging and Discharging Power in pu
ess_charging = {node: env_config_dict['p_ch_max'] for node in env_config_dict['ess_nodes']}
ess_discharging = {node: 0 for node in env_config_dict['ess_nodes']}

# Initial Energy Content of Energy Storage Systems in pu
initial_ess_energy = {node: env_config_dict['e_max']/2 for node in env_config_dict['ess_nodes']}  

# Execute the power flow solver
results = power_flow_solver(network_data, active_power_demand, reactive_power_demand, power_reduction, pv_power, pv_reactive_power, ess_charging, ess_discharging, initial_ess_energy)

# Log the results
logger.info('Power flow results: %s', results)

# Plot the results if necessary
plot_power_flow_results(results)
