import matplotlib.pyplot as plt
from MADRL.environments.flex_provision.flexibility_provision_env import FlexibilityProvisionEnv
import numpy as np
import yaml
from OPF.constants import S_NOM
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to plot results
def plot_results(active_demand, reactive_demand, pv_power, voltages, prices, ess_energy, rewards):
    timesteps = range(len(active_demand))

    # Active Power Demand
    plt.figure(figsize=(10, 6))
    for bus_id in active_demand[0].keys():
        plt.plot(timesteps, [ad[bus_id] for ad in active_demand], label=f'Bus {bus_id}')
    plt.title('Active Power Demand Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Active Power Demand (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_env_results/Active_Power_Demand.png')
    plt.close()

    # Reactive Power Demand
    plt.figure(figsize=(10, 6))
    for bus_id in reactive_demand[0].keys():
        plt.plot(timesteps, [rd[bus_id] for rd in reactive_demand], label=f'Bus {bus_id}')
    plt.title('Reactive Power Demand Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Reactive Power Demand (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/run_env_results/Reactive_Power_Demand.png')
    plt.close()

    # PV Power
    plt.figure(figsize=(10, 6))
    for i, bus_id in enumerate(pv_power[0].keys()):
        plt.plot(timesteps, [pv[bus_id] for pv in pv_power], label=f'PV {bus_id}')
    plt.title('PV Power Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('PV Power (p.u.)')
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
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.savefig('MADRL/plots/run_env_results/Price.png')
    plt.close()

    # ESS Energy
    plt.figure(figsize=(10, 6))
    for ess_id in ess_energy[0].keys():
        plt.plot(timesteps, [e[ess_id] for e in ess_energy], label=f'ESS {ess_id}')
    plt.title('ESS Energy Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Energy (p.u.)')
    plt.grid(True)
    plt.legend()
    plt.savefig('MADRL/plots/run_env_results/ESS_Energy.png')
    plt.close()

    # Cumulative Reward
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, 'o-')
    plt.title('Cumulative Reward Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.savefig('MADRL/plots/run_env_results/Cumulative_Reward.png')
    plt.close()