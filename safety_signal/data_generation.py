import numpy as np
import pandas as pd
import yaml
import os
from utils.create_net import create_network
from utils.pf import power_flow_solver_simplified 

# load env args
with open("./madrl/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"]

# Number of scenarios to generate
num_scenarios = 1000

# Load network data
network_data = create_network()

# Base net power at each bus
P_base = network_data['active_power_demand']
Q_base = network_data['reactive_power_demand']

# Initialize lists to store data
input_data = []
output_data = []

# Variation percentage (±30%)
variation_percentage = 0.3

# Loop to generate scenarios
for scenario in range(num_scenarios):
    # Generate random net active and reactive power values
    P_net = {bus: P_base[bus] * (1 + np.random.uniform(-variation_percentage, variation_percentage))
             for bus in network_data['bus_numbers']}
    Q_net = {bus: Q_base[bus] * (1 + np.random.uniform(-variation_percentage, variation_percentage))
             for bus in network_data['bus_numbers']}

    try:
        # Run power flow solver
        results = power_flow_solver_simplified(network_data, P_net, Q_net)

        # Extract voltages
        voltages = results['Voltages']

        # Prepare input and output vectors
        input_vector = []
        output_vector = []
        for bus in network_data['bus_numbers']:
            input_vector.extend([P_net[bus], Q_net[bus]])  # Net powers
            output_vector.append(voltages[bus])  # Voltage magnitude

        # Append to data lists
        input_data.append(input_vector)
        output_data.append(output_vector)

    except Exception as e:
        print(f"Scenario {scenario}: Solver failed - {e}")
        continue  # Skip this scenario

# Convert to DataFrames
input_columns = []
output_columns = []
for bus in network_data['bus_numbers']:
    input_columns.extend([f'P_net_{bus}', f'Q_net_{bus}'])
    output_columns.append(f'Voltage_{bus}')

input_df = pd.DataFrame(input_data, columns=input_columns)
output_df = pd.DataFrame(output_data, columns=output_columns)

input_file_name = "net_power_inputs.csv"
output_file_name = "bus_voltages_outputs.csv"

# Construct full file paths
input_file_path = os.path.join(data_path, input_file_name)
output_file_path = os.path.join(data_path, output_file_name)

# Save to CSV files
input_df.to_csv(input_file_path, index=False)
output_df.to_csv(output_file_path, index=False)
