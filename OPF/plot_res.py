import matplotlib.pyplot as plt
import numpy as np
from OPF.constants import S_NOM

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
        plt.plot(times, [results['Power Reduction'][t][b] * S_NOM for t in times], marker='o', label=f'Building {b}')
    plt.title('Power Reduction by Buildings')
    plt.xlabel('Time')
    plt.ylabel('Power Reduction (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/Power_Reduction_by_Buildings.png')

    # Plot Reactive Power Provided by PVs
    plt.figure(figsize=(10, 6))
    for pv in pv_ids:
        plt.plot(times, [results['PV Reactive Power'][t][pv] * S_NOM for t in times], marker='o', label=f'PV {pv}')
    plt.title('Reactive Power Provided by PVs')
    plt.xlabel('Time')
    plt.ylabel('Reactive Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/Reactive_Power_Provided_by_PVs.png')
    plt.close()

    # Plot ESS Charging
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['ESS Charging'][t][k] * S_NOM for t in times], marker='o', label=f'ESS {k}')
    plt.title('ESS Charging Power')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/ESS_Charging_Power.png')

    # Plot ESS Discharging
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['ESS Discharging'][t][k] * S_NOM for t in times], marker='o', label=f'ESS {k}')
    plt.title('ESS Discharging Power')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/ESS_Discharging_Power.png')

    # Plot ESS Charging Indicator
    plt.figure(figsize=(10, 6))
    for k in ess_ids:
        plt.plot(times, [results['Charging Indicator'][t][k] for t in times], marker='o', label=f'ESS {k}')
    plt.title('ESS Charging Indicator')
    plt.xlabel('Time')
    plt.ylabel('Charging Indicator (0 or 1)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/ESS_Charging_Indicator.png')
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
    plt.savefig('OPF/plots/Voltage_at_Buses.png')

    # Plot Active Power Load
    plt.figure(figsize=(10, 6))
    for n in buses:
        plt.plot(times, [results['Active Power Load'][t][n] * S_NOM for t in times], marker='o', label=f'Bus {n}')
    plt.title('Active Power Load')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/Active_Power_Load.png')
    plt.close()

    # Plot Reactive Power Load
    plt.figure(figsize=(10, 6))
    for n in buses:
        plt.plot(times, [results['Reactive Power Load'][t][n] * S_NOM for t in times], marker='o', label=f'Bus {n}')
    plt.title('Reactive Power Load')
    plt.xlabel('Time')
    plt.ylabel('Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/Reactive_Power_Load.png')
    plt.close()

    # Plot Active Power from PVs
    plt.figure(figsize=(10, 6))
    for pv in pv_ids:
        plt.plot(times, [results['PV Active Power'][t][pv] * S_NOM for t in times], marker='o', label=f'PV {pv}')
    plt.title('Active Power from PVs')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/Active_Power_from_PVs.png')
    plt.close()

    # Plot Active Power Flow
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [results['Active Power Flow'][t][l] * S_NOM for t in times], marker='o', label=f'Line {l}')
    plt.title('Active Power Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/Active_Power_Flow_on_Lines.png')

    # Plot Reactive Power Flow
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [results['Reactive Power Flow'][t][l] * S_NOM for t in times], marker='o', label=f'Line {l}')
    plt.title('Reactive Power Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Power (kVar)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/Reactive_Power_Flow_on_Lines.png')

    # Plot Currents on Lines
    plt.figure(figsize=(10, 6))
    for l in lines:
        plt.plot(times, [np.sqrt(results['Current Squared'][t][l]) for t in times], marker='o', label=f'Line {l}')
    plt.title('Current Flow on Lines')
    plt.xlabel('Time')
    plt.ylabel('Current (pu)')
    plt.grid(True)
    plt.legend()
    plt.savefig('OPF/plots/Current_Flow_on_Lines.png')