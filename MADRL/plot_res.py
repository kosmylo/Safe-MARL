import matplotlib.pyplot as plt
import numpy as np
from OPF.constants import S_NOM

def plot_power_flow_results(results):
    # Plot Voltage Profiles at Buses
    plt.figure(figsize=(10, 6))
    voltages = results['Voltages']
    buses = sorted(voltages.keys())
    # Since voltages are stored squared, take the square root for plotting
    plt.plot(buses, [np.sqrt(voltages[bus]) for bus in buses], 'o-', label='Voltage (p.u.)')
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
    lines = sorted(currents.keys())
    # Currents may be stored directly as values or squared
    plt.plot(lines, [np.sqrt(currents[line]) for line in lines], 'o-', label='Current (p.u.)')
    plt.title('Current Flow on Lines')
    plt.xlabel('Line')
    plt.ylabel('Current (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig('MADRL/plots/pf_results/Current_Flow.png')
    plt.close()

    # Plot Active and Reactive Power Flows on Lines
    plt.figure(figsize=(10, 6))
    power_flows = results['Power Flows']
    for line, (p_flow, q_flow) in power_flows.items():
        plt.plot([line], [p_flow * S_NOM], 'o', label=f'{line} P (kW)')
        plt.plot([line], [q_flow * S_NOM], 'x', label=f'{line} Q (kVar)')

    plt.title('Power Flows on Lines')
    plt.xlabel('Line')
    plt.ylabel('Power (kW, kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig('MADRL/plots/pf_results/Power_Flows.png')
    plt.close()

    # Plot Next Energy State of ESS
    plt.figure(figsize=(10, 6))
    if 'Next ESS Energy' in results:
        ess_energy = results['Next ESS Energy']
        ess_ids = sorted(ess_energy.keys())
        plt.bar(ess_ids, [ess_energy[ess] * S_NOM for ess in ess_ids], color='blue', label='Next ESS Energy (kWh)')
        plt.title('Next Energy State of ESS')
        plt.xlabel('ESS ID')
        plt.ylabel('Energy (kWh)')
        plt.grid(True)
        plt.legend()
        plt.savefig('MADRL/plots/pf_results/Next_ESS_Energy.png')
        plt.close()
