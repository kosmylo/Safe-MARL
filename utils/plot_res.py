import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

# load env args
with open("./madrl/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]

def ensure_directory_exists(path):
    """Ensure that the directory exists, if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)

def plot_optimization_results(results):

    save_path = 'plots/run_opf_results/'
    ensure_directory_exists(save_path)

    # Extract time periods from the results
    times = list(results['Power Reduction'].keys())
    buildings = list(results['Power Reduction'][times[0]].keys())
    ess_ids = list(results['ESS Charging'][times[0]].keys())
    pv_ids = list(results['PV Reactive Power'][times[0]].keys())
    buses = list(results['Voltage Squared'][times[0]].keys())
    lines = list(results['Active Power Flow'][times[0]].keys())

    # Plot Power Reduction
    plt.figure(figsize=(10, 6))
    for b in buildings:
        plt.plot(times, [results['Power Reduction'][t][b] * env_config_dict['s_nom'] for t in times], label=f'Building {b}')
    plt.title('Power Reduction by Buildings')
    plt.xlabel('Time')
    plt.ylabel('Power Reduction (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Power_Reduction_by_Buildings.png'))
    plt.close()

    # Plot Reactive Power Provided by PVs
    plt.figure(figsize=(10, 6))
    for pv in pv_ids:
        plt.plot(times, [results['PV Reactive Power'][t][pv] * env_config_dict['s_nom'] for t in times], label=f'PV {pv}')
    plt.title('Reactive Power Provided by PVs')
    plt.xlabel('Time')
    plt.ylabel('Reactive Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Reactive_Power_Provided_by_PVs.png'))
    plt.close()

    # Plot ESS Charging
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['ESS Charging'][t][k] * env_config_dict['s_nom'] for t in times], label=f'ESS {k}')
    plt.title('ESS Charging Power')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Charging_Power.png'))
    plt.close()

    # Plot ESS Discharging
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['ESS Discharging'][t][k] * env_config_dict['s_nom'] for t in times], label=f'ESS {k}')
    plt.title('ESS Discharging Power')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Discharging_Power.png'))
    plt.close()

    # Plot ESS Charging Indicator
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['Charging Indicator'][t][k] for t in times], label=f'ESS {k}')
    plt.title('ESS Charging Indicator')
    plt.xlabel('Time')
    plt.ylabel('Charging Indicator (0 or 1)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Charging_Indicator.png'))    
    plt.close()

    # Plot ESS Energy
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['ESS Energy'][t][k] * env_config_dict['s_nom'] for t in times], label=f'ESS {k}')
    plt.title('ESS Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Energy.png'))  
    plt.close()

    # Plot Voltage Profiles
    plt.figure(figsize=(10, 6))
    for n in buses:
        plt.plot(times, [np.sqrt(results['Voltage Squared'][t][n]) for t in times], label=f'Bus {n}')
    plt.title('Voltage at Buses')
    plt.xlabel('Time')
    plt.ylabel('Voltage (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Voltage_at_Buses.png'))  
    plt.close()

    # Plot Active Power Load
    plt.figure(figsize=(10, 6))
    for n in buses:
        plt.plot(times, [results['Active Power Load'][t][n] * env_config_dict['s_nom'] for t in times], label=f'Bus {n}')
    plt.title('Active Power Load')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Active_Power_Load.png'))  
    plt.close()

    # Plot Reactive Power Load
    plt.figure(figsize=(10, 6))
    for n in buses:
        plt.plot(times, [results['Reactive Power Load'][t][n] * env_config_dict['s_nom'] for t in times], label=f'Bus {n}')
    plt.title('Reactive Power Load')
    plt.xlabel('Time')
    plt.ylabel('Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Reactive_Power_Load.png'))  
    plt.close()

    # Plot Active Power from PVs
    plt.figure(figsize=(10, 6))
    for pv in pv_ids:
        plt.plot(times, [results['PV Active Power'][t][pv] * env_config_dict['s_nom'] for t in times], label=f'PV {pv}')
    plt.title('Active Power from PVs')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Active_Power_from_PVs.png'))  
    plt.close()

    # Plot Active Power Flow
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [results['Active Power Flow'][t][l] * env_config_dict['s_nom'] for t in times], label=f'Line {l}')
    plt.title('Active Power Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Active_Power_Flow_on_Lines.png'))  
    plt.close()

    # Plot Reactive Power Flow
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [results['Reactive Power Flow'][t][l] * env_config_dict['s_nom'] for t in times], label=f'Line {l}')
    plt.title('Reactive Power Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Reactive_Power_Flow_on_Lines.png'))  
    plt.close()

    # Plot Currents on Lines
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [np.sqrt(results['Current Squared'][t][l]) for t in times], label=f'Line {l}')
    plt.title('Current Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Current (pu)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Current_Flow_on_Lines.png')) 
    plt.close()

def plot_power_flow_results(results):

    save_path = 'plots/run_pf_results/'
    ensure_directory_exists(save_path)
    
    # Plot Voltage Profiles at Buses
    plt.figure(figsize=(10, 6))
    voltages = results['Voltages']
    buses = voltages.keys()
    voltage_values = [voltage for voltage in voltages.values()]
    plt.plot(buses, voltage_values, 'o-', label='Voltage (p.u.)')
    plt.title('Voltage Profile Across Buses')
    plt.xlabel('Bus Number')
    plt.ylabel('Voltage (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Voltage_Profile.png'))
    plt.close()

    # Plot Currents on Lines
    plt.figure(figsize=(10, 6))
    currents = results['Currents']
    lines = [f'{line[0]}-{line[1]}' for line in currents.keys()]
    current_values = [current for current in currents.values()]
    plt.plot(lines, current_values, 'o-', label='Current on Lines')
    plt.title('Current Flow on Lines')
    plt.xlabel('Line')
    plt.ylabel('Current (p.u.)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Current_Flow.png'))
    plt.close()

    # Plot Active and Reactive Power Flows on Lines
    plt.figure(figsize=(10, 6))
    power_flows = results['Power Flows']
    lines = [f'{line[0]}-{line[1]}' for line in power_flows.keys()]
    p_flows = [p_flow * env_config_dict['s_nom'] for p_flow, q_flow in power_flows.values()]
    q_flows = [q_flow * env_config_dict['s_nom'] for p_flow, q_flow in power_flows.values()]
    plt.plot(lines, p_flows, 'o-', label='Active Power (kW)')
    plt.plot(lines, q_flows, 'x-', label='Reactive Power (kVar)')
    plt.title('Power Flows on Lines')
    plt.xlabel('Line')
    plt.ylabel('Power (kW, kVar)')
    plt.xticks(rotation=45)  # Rotate labels to avoid collision
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to give enough space for label rotation
    plt.savefig(os.path.join(save_path, 'Power_Flows.png'))
    plt.close()

    # Plot Next Energy State of ESS
    plt.figure(figsize=(10, 6))
    ess_energy = results['Next ESS Energy']
    ess_ids = ess_energy.keys()
    plt.bar(ess_ids, [ess_energy[ess] * env_config_dict['s_nom'] for ess in ess_ids], color='blue', label='Next ESS Energy (kWh)')
    plt.title('Next Energy State of ESS')
    plt.xlabel('ESS ID')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Next_ESS_Energy.png'))
    plt.close()

# Function to plot environment results
def plot_environment_results(active_demand, reactive_demand, pv_power, voltages, 
                             prices, ess_energy, rewards, percentage_reduction, ess_charging, ess_discharging, q_pv):
    
    save_path = 'plots/run_env_results/'
    ensure_directory_exists(save_path)
    
    timesteps = range(len(active_demand))

    # Active Power Demand
    plt.figure(figsize=(10, 6))
    for bus_id in active_demand[0].keys():
        plt.plot(timesteps, [ad[bus_id] * env_config_dict['s_nom'] for ad in active_demand], label=f'Bus {bus_id}')
    plt.title('Active Power Demand Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Active Power Demand (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Active_Power_Demand.png'))
    plt.close()

    # Reactive Power Demand
    plt.figure(figsize=(10, 6))
    for bus_id in reactive_demand[0].keys():
        plt.plot(timesteps, [rd[bus_id] * env_config_dict['s_nom'] for rd in reactive_demand], label=f'Bus {bus_id}')
    plt.title('Reactive Power Demand Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Reactive Power Demand (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Reactive_Power_Demand.png'))
    plt.close()

    # PV Power
    plt.figure(figsize=(10, 6))
    for i, bus_id in enumerate(pv_power[0].keys()):
        plt.plot(timesteps, [pv[bus_id] * env_config_dict['s_nom'] for pv in pv_power], label=f'PV {bus_id}')
    plt.title('PV Power Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('PV Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'PV_Power.png'))
    plt.close()

    # Voltage Profile
    plt.figure(figsize=(10, 6))
    for bus_id in voltages[0].keys():
        plt.plot(timesteps, [v[bus_id] for v in voltages], label=f'Bus {bus_id}')
    plt.title('Voltage Profile Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Voltage_Profile.png'))
    plt.close()

    # Prices
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, prices, label='Price')
    plt.title('Price Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Price (Euros/kWh)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Price.png'))
    plt.close()

    # ESS Energy
    plt.figure(figsize=(10, 6))
    for ess_id in ess_energy[0].keys():
        plt.plot(timesteps, [e[ess_id] * env_config_dict['s_nom'] for e in ess_energy], label=f'ESS {ess_id}')
    plt.title('ESS Energy Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Energy.png'))
    plt.close()

    # Cumulative Reward
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, 'o-')
    plt.title('Cumulative Reward Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward (Euros)')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Cumulative_Reward.png'))
    plt.close()

    # Plot Percentage Reduction
    plt.figure(figsize=(10, 6))
    for building in percentage_reduction[0].keys():
        plt.plot(timesteps, [pr[building] * env_config_dict['s_nom'] * active_demand[t][building]for t, pr in enumerate(percentage_reduction)], 
                 label=f'Building {building}')
    plt.title('Power Reduction Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Power Reduction (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Power_Reduction.png'))
    plt.close()

    # Plot ESS Charging
    plt.figure(figsize=(10, 6))
    for ess in ess_charging[0].keys():
        plt.plot(timesteps, [ch[ess] * env_config_dict['s_nom'] for ch in ess_charging], label=f'ESS {ess} Charging')
    plt.title('ESS Charging Power Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Charging Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Charging.png'))
    plt.close()

    # Plot ESS Discharging
    plt.figure(figsize=(10, 6))
    for ess in ess_discharging[0].keys():
        plt.plot(timesteps, [dch[ess] * env_config_dict['s_nom'] for dch in ess_discharging], label=f'ESS {ess} Discharging')
    plt.title('ESS Discharging Power Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Discharging Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Discharging.png'))
    plt.close()

    # Plot Reactive Power of PVs
    plt.figure(figsize=(10, 6))
    for pv in q_pv[0].keys():
        plt.plot(timesteps, [q[pv] * env_config_dict['s_nom'] for q in q_pv], label=f'PV {pv} Reactive Power')
    plt.title('PV Reactive Power Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Reactive Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'PV_Reactive_Power.png'))
    plt.close()

def plot_training_metrics(rewards, policy_losses, value_losses):

    save_path = 'plots/train_results/'
    ensure_directory_exists(save_path)

    # Plot average reward per episode
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards, 'o-', label='Average Reward per Episode')
    plt.title('Average Reward During Training')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Euros)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Average_Reward_During_Training.png'))
    plt.close()

    # Plot policy loss per episode
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(policy_losses)), policy_losses, 'o-', label='Policy Loss per Episode')
    plt.title('Policy Loss During Training')
    plt.xlabel('Episode')
    plt.ylabel('Policy Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Policy_Loss_During_Training.png'))
    plt.close()

    # Plot value loss per episode
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(value_losses)), value_losses, 'o-', label='Value Loss per Episode')
    plt.title('Value Loss During Training')
    plt.xlabel('Episode')
    plt.ylabel('Value Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Value_Loss_During_Training.png'))
    plt.close()

def plot_testing_results(record):
    """Plot the testing results from the test phase."""

    save_path = 'plots/test_results/'
    ensure_directory_exists(save_path)
    
    timesteps = range(len(record['pv_active']))

    # Plot Active Power from PVs
    plt.figure(figsize=(10, 6))
    for i, pv in enumerate(record['pv_active'][0]):
        plt.plot(timesteps, [pv_val[i] * env_config_dict['s_nom'] for pv_val in record['pv_active']], label=f'PV {i}')
    plt.title('Active Power from PVs Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Active Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Active_Power_from_PVs.png'))
    plt.close()

    # Plot Reactive Power from PVs
    plt.figure(figsize=(10, 6))
    for i, pv in enumerate(record['pv_reactive'][0]):
        plt.plot(timesteps, [q_pv_val[i] * env_config_dict['s_nom'] for q_pv_val in record['pv_reactive']], label=f'PV {i}')
    plt.title('Reactive Power from PVs Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Reactive Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Reactive_Power_from_PVs.png'))
    plt.close()

    # Plot Active Power Demand at Buses
    plt.figure(figsize=(10, 6))
    for i, bus in enumerate(record['bus_active'][0]):
        plt.plot(timesteps, [bus_active[i] * env_config_dict['s_nom'] for bus_active in record['bus_active']], label=f'Bus {i}')
    plt.title('Active Power Demand at Buses Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Active Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Active_Power_Demand_at_Buses.png'))
    plt.close()

    # Plot Reactive Power Demand at Buses
    plt.figure(figsize=(10, 6))
    for i, bus in enumerate(record['bus_reactive'][0]):
        plt.plot(timesteps, [bus_reactive[i] * env_config_dict['s_nom'] for bus_reactive in record['bus_reactive']], label=f'Bus {i}')
    plt.title('Reactive Power Demand at Buses Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Reactive Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Reactive_Power_Demand_at_Buses.png'))
    plt.close()

    # Plot Voltage at Buses (in p.u.)
    plt.figure(figsize=(10, 6))
    for i, bus in enumerate(record['bus_voltage'][0]):
        plt.plot(timesteps, [bus_voltage[i] for bus_voltage in record['bus_voltage']], label=f'Bus {i}')
    plt.title('Voltage at Buses Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Voltage_at_Buses.png'))
    plt.close()

    # Plot ESS Energy (converted to kWh)
    plt.figure(figsize=(10, 6))
    for i, ess in enumerate(record['ess_energy'][0]):
        plt.plot(timesteps, [ess_energy[i] * env_config_dict['s_nom'] for ess_energy in record['ess_energy']], label=f'ESS {i}')
    plt.title('ESS Energy Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Energy.png'))
    plt.close()

    # Plot Power Reduction (converted to kW)
    plt.figure(figsize=(10, 6))
    for i, building in enumerate(record['power_reduction'][0]):
        plt.plot(timesteps, [power_reduction[i] * env_config_dict['s_nom'] for power_reduction in record['power_reduction']], label=f'Building {i}')
    plt.title('Power Reduction Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Power Reduction (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Power_Reduction.png'))
    plt.close()

    # Plot ESS Charging Power (converted to kW)
    plt.figure(figsize=(10, 6))
    for i, ess in enumerate(record['ess_charging'][0]):
        plt.plot(timesteps, [ess_charging[i] * env_config_dict['s_nom'] for ess_charging in record['ess_charging']], label=f'ESS {i} Charging')
    plt.title('ESS Charging Power Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Charging Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Charging.png'))
    plt.close()

    # Plot ESS Discharging Power (converted to kW)
    plt.figure(figsize=(10, 6))
    for i, ess in enumerate(record['ess_discharging'][0]):
        plt.plot(timesteps, [ess_discharging[i] * env_config_dict['s_nom'] for ess_discharging in record['ess_discharging']], label=f'ESS {i} Discharging')
    plt.title('ESS Discharging Power Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Discharging Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'ESS_Discharging.png'))
    plt.close()