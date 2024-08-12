import matplotlib.pyplot as plt
import numpy as np
import yaml

# load env args
with open("./MADRL/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]

def plot_optimization_results(results):

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
        plt.plot(times, [results['Power Reduction'][t][b] * env_config_dict['s_nom'] for t in times], marker='o', label=f'Building {b}')
    plt.title('Power Reduction by Buildings')
    plt.xlabel('Time')
    plt.ylabel('Power Reduction (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Power_Reduction_by_Buildings.png')

    # Plot Reactive Power Provided by PVs
    plt.figure(figsize=(10, 6))
    for pv in pv_ids:
        plt.plot(times, [results['PV Reactive Power'][t][pv] * env_config_dict['s_nom'] for t in times], marker='o', label=f'PV {pv}')
    plt.title('Reactive Power Provided by PVs')
    plt.xlabel('Time')
    plt.ylabel('Reactive Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Reactive_Power_Provided_by_PVs.png')
    plt.close()

    # Plot ESS Charging
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['ESS Charging'][t][k] * env_config_dict['s_nom'] for t in times], marker='o', label=f'ESS {k}')
    plt.title('ESS Charging Power')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/ESS_Charging_Power.png')

    # Plot ESS Discharging
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['ESS Discharging'][t][k] * env_config_dict['s_nom'] for t in times], marker='o', label=f'ESS {k}')
    plt.title('ESS Discharging Power')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/ESS_Discharging_Power.png')

    # Plot ESS Charging Indicator
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['Charging Indicator'][t][k] for t in times], marker='o', label=f'ESS {k}')
    plt.title('ESS Charging Indicator')
    plt.xlabel('Time')
    plt.ylabel('Charging Indicator (0 or 1)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/ESS_Charging_Indicator.png')
    plt.close()

    # Plot ESS Energy
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['ESS Energy'][t][k] * env_config_dict['s_nom'] for t in times], marker='o', label=f'ESS {k}')
    plt.title('ESS Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/ESS_Energy.png')
    plt.close()

    # Plot Voltage Profiles
    plt.figure(figsize=(10, 6))
    for n in buses:
        plt.plot(times, [np.sqrt(results['Voltage Squared'][t][n]) for t in times], marker='o', label=f'Bus {n}')
    plt.title('Voltage at Buses')
    plt.xlabel('Time')
    plt.ylabel('Voltage (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Voltage_at_Buses.png')

    # Plot Active Power Load
    plt.figure(figsize=(10, 6))
    for n in buses:
        plt.plot(times, [results['Active Power Load'][t][n] * env_config_dict['s_nom'] for t in times], marker='o', label=f'Bus {n}')
    plt.title('Active Power Load')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Active_Power_Load.png')
    plt.close()

    # Plot Reactive Power Load
    plt.figure(figsize=(10, 6))
    for n in buses:
        plt.plot(times, [results['Reactive Power Load'][t][n] * env_config_dict['s_nom'] for t in times], marker='o', label=f'Bus {n}')
    plt.title('Reactive Power Load')
    plt.xlabel('Time')
    plt.ylabel('Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Reactive_Power_Load.png')
    plt.close()

    # Plot Active Power from PVs
    plt.figure(figsize=(10, 6))
    for pv in pv_ids:
        plt.plot(times, [results['PV Active Power'][t][pv] * env_config_dict['s_nom'] for t in times], marker='o', label=f'PV {pv}')
    plt.title('Active Power from PVs')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Active_Power_from_PVs.png')
    plt.close()

    # Plot Active Power Flow
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [results['Active Power Flow'][t][l] * env_config_dict['s_nom'] for t in times], marker='o', label=f'Line {l}')
    plt.title('Active Power Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Active_Power_Flow_on_Lines.png')

    # Plot Reactive Power Flow
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [results['Reactive Power Flow'][t][l] * env_config_dict['s_nom'] for t in times], marker='o', label=f'Line {l}')
    plt.title('Reactive Power Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Reactive_Power_Flow_on_Lines.png')

    # Plot Currents on Lines
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [np.sqrt(results['Current Squared'][t][l]) for t in times], marker='o', label=f'Line {l}')
    plt.title('Current Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Current (pu)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_opf_results/Current_Flow_on_Lines.png')

def plot_power_flow_results(results):
    
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
    plt.savefig('plots/run_pf_results/Voltage_Profile.png')
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
    plt.savefig('plots/run_pf_results/Current_Flow.png')
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
    plt.savefig('plots/run_pf_results/Power_Flows.png')
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
    plt.savefig('plots/run_pf_results/Next_ESS_Energy.png')
    plt.close()

# Function to plot environment results
def plot_environment_results(active_demand, reactive_demand, pv_power, voltages, prices, ess_energy, rewards):
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
    plt.savefig('plots/run_env_results/Active_Power_Demand.png')
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
    plt.savefig('plots/run_env_results/Reactive_Power_Demand.png')
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
    plt.savefig('plots/run_env_results/PV_Power.png')
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
    plt.savefig('plots/run_env_results/Voltage_Profile.png')
    plt.close()

    # Prices
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, prices, label='Price')
    plt.title('Price Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Price (Euros/kWh)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_env_results/Price.png')
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
    plt.savefig('plots/run_env_results/ESS_Energy.png')
    plt.close()

    # Cumulative Reward
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, 'o-')
    plt.title('Cumulative Reward Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward (Euros)')
    plt.grid(True)
    plt.savefig('plots/run_env_results/Cumulative_Reward.png')
    plt.close()
