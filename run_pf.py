from utils.create_net import create_network
from OPF.constants import BUILDINGS, PV_NODES, ESS_NODES, PV_CAPACITY, E_MAX, P_CH_MAX, P_DIS_MAX, MAX_POWER_REDUCTION_PERCENT
from MADRL.pf import power_flow_solver
from MADRL.plot_res import plot_power_flow_results
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the network data
network_data = create_network()

# Example Active and Reactive Power Demand in pu
active_power_demand = {node: 0.1 for node in network_data['bus_numbers']}  # 10% of S_NOM
reactive_power_demand = {node: 0.05 for node in network_data['bus_numbers']}  # 5% of S_NOM

# Active Power Reduction capabilities at some buildings (Demand Response) in pu
power_reduction = {node: active_power_demand[node] * MAX_POWER_REDUCTION_PERCENT for node in BUILDINGS}

# Photovoltaic Power Generation at certain buses in pu
pv_power = {node: PV_CAPACITY for node in PV_NODES}

# Photovoltaic Reactive Power Generation (if applicable) in pu
pv_reactive_power = {node: pv_power[node] * 0.1 for node in PV_NODES}  # Assuming 10% of PV power can be reactive

# Energy Storage System Charging and Discharging Power in pu
ess_charging = {node: P_CH_MAX for node in ESS_NODES}
ess_discharging = {node: 0 for node in ESS_NODES}

# Initial Energy Content of Energy Storage Systems in pu
initial_ess_energy = {node: E_MAX / 2 for node in ESS_NODES}  # Initial state halfway to E_MAX

# Execute the power flow solver
results = power_flow_solver(network_data, active_power_demand, reactive_power_demand, power_reduction, pv_power, pv_reactive_power, ess_charging, ess_discharging, initial_ess_energy)

# Log the results
logger.info('Power flow results: %s', results)

# Plot the results if necessary
plot_power_flow_results(results)
