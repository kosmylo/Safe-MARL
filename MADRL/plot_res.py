import matplotlib.pyplot as plt
import numpy as np
from OPF.constants import S_NOM

def plot_power_flow_results(results):
    # Plot Voltage Profiles at Buses
    plt.figure(figsize=(10, 6))
    voltages = results['Voltages']
    buses = voltages.keys()
    voltage_values = [np.sqrt(voltage) for voltage in voltages.values()]
    plt.plot(buses, voltage_values, 'o-', label='Voltage (p.u.)')
    plt.title('Voltage Profile Across Buses')
    plt.xlabel('Bus Number')
    plt.ylabel('Voltage (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig('MADRL/plots/pf_results/Voltage_Profile.png')
    plt.close()

    # Plot Currents on Lines
    plt.figure(figsize=(10, 6))
    currents = results['Currents']
    lines = [f'{line[0]}-{line[1]}' for line in currents.keys()]
    current_values = [np.sqrt(current) for current in currents.values()]
    plt.plot(lines, current_values, 'o-', label='Current on Lines')
    plt.title('Current Flow on Lines')
    plt.xlabel('Line')
    plt.ylabel('Current (p.u.)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('MADRL/plots/pf_results/Current_Flow.png')
    plt.close()

    # Plot Active and Reactive Power Flows on Lines
    plt.figure(figsize=(10, 6))
    power_flows = results['Power Flows']
    lines = [f'{line[0]}-{line[1]}' for line in power_flows.keys()]
    p_flows = [p_flow * S_NOM for p_flow, q_flow in power_flows.values()]
    q_flows = [q_flow * S_NOM for p_flow, q_flow in power_flows.values()]
    plt.plot(lines, p_flows, 'o-', label='Active Power (kW)')
    plt.plot(lines, q_flows, 'x-', label='Reactive Power (kVar)')
    plt.title('Power Flows on Lines')
    plt.xlabel('Line')
    plt.ylabel('Power (kW, kVar)')
    plt.xticks(rotation=45)  # Rotate labels to avoid collision
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to give enough space for label rotation
    plt.savefig('MADRL/plots/pf_results/Power_Flows.png')
    plt.close()

    # Plot Next Energy State of ESS
    plt.figure(figsize=(10, 6))
    ess_energy = results['Next ESS Energy']
    ess_ids = ess_energy.keys()
    plt.bar(ess_ids, [ess_energy[ess] * S_NOM for ess in ess_ids], color='blue', label='Next ESS Energy (kWh)')
    plt.title('Next Energy State of ESS')
    plt.xlabel('ESS ID')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.savefig('MADRL/plots/pf_results/Next_ESS_Energy.png')
    plt.close()
