import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import yaml
import numpy as np

# load env args
with open("./madrl/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]

def power_flow_solver(network_data, active_power_demand, reactive_power_demand, power_reduction, pv_active_power, pv_reactive_power, ess_charging, ess_discharging, initial_ess_energy):
    
    # Define the model
    model = pyo.ConcreteModel()

    # Define Sets
    model.N = pyo.Set(initialize=network_data['bus_numbers'])  # Buses
    model.L = pyo.Set(initialize=network_data['line_connections'])  # Lines
    model.B = pyo.Set(initialize=network_data['buildings'])  # Buildings with flexibility
    model.G = pyo.Set(initialize=network_data['PVs_at_buildings'])  # PVs
    model.K = pyo.Set(initialize=network_data['ESSs_at_buildings'])  # ESSs

    # Define Parameters
    model.R = pyo.Param(model.L, initialize=network_data['line_resistances'])  # Resistance of each line
    model.X = pyo.Param(model.L, initialize=network_data['line_reactances'])  # Reactance of each line
    model.Pload = pyo.Param(model.N, initialize=active_power_demand)  # Active power demand at each bus
    model.Qload = pyo.Param(model.N, initialize=reactive_power_demand)  # Reactive power demand at each bus
    model.Pred = pyo.Param(model.B, initialize=power_reduction)  # Active power reduction at buildings
    model.Ppv = pyo.Param(model.G, initialize=pv_active_power)  # PV power generation
    model.Qpv = pyo.Param(model.G, initialize=pv_reactive_power)  # PV reactive power generation
    model.Pesc = pyo.Param(model.K, initialize=ess_charging)  # Power charged to ESS
    model.Pesd = pyo.Param(model.K, initialize=ess_discharging)  # Power discharged from ESS
    model.eta_ch = pyo.Param(model.K, initialize=env_config_dict['eta_ch']) # Charging efficiency of ESS
    model.eta_dis = pyo.Param(model.K, initialize=env_config_dict['eta_dis']) # Discharging efficiency of ESS
    model.E_init = pyo.Param(model.K, initialize=initial_ess_energy)  # Initial ESS energy state

    # Define Variables
    model.Vsqr = pyo.Var(model.N, within=pyo.NonNegativeReals)  # Voltage squared at each bus
    model.Pl = pyo.Var(model.L, within=pyo.Reals)  # Active power flow on each line
    model.Ql = pyo.Var(model.L, within=pyo.Reals)  # Reactive power flow on each line
    model.Isqr = pyo.Var(model.L, within=pyo.NonNegativeReals)  # Current squared on each line
    model.E_next = pyo.Var(model.K, within=pyo.NonNegativeReals)  # ESS energy content for the next timestep
    model.Ps = pyo.Var(model.N, within=pyo.Reals)  # Active power generation or absorption at each bus
    model.Qs = pyo.Var(model.N, within=pyo.Reals)  # Reactive power generation or absorption at each bus

    # Slack Bus Identification and Configuration
    for n in model.N:
        if network_data['bus_types'][n] == 1:  # Check if the bus is a slack bus
            model.Vsqr[n].fix(1)  # Fix voltage at the slack bus to 1 per unit
        else:
            model.Ps[n].fix(0)  # Fix active power at non-slack buses to zero
            model.Qs[n].fix(0)  # Fix reactive power at non-slack buses to zero
    
    # Constraints
    def active_power_balance_rule(model, n):
        # Calculates power balance including generation, demand, and storage effects
        return (sum(model.Pl[i, j] for (i, j) in model.L if j == n) - 
               sum(model.Pl[i, j] + model.R[i, j] * model.Isqr[i, j] for (i, j) in model.L if i == n) + 
               model.Ps[n] - 
               model.Pload[n] + (model.Pred[n] if n in model.Pred else 0) + 
               (model.Ppv[n] if n in model.Ppv else 0) - 
               (model.Pesc[n] if n in model.K else 0) + 
               (model.Pesd[n] if n in model.K else 0) == 0)
    model.active_power_balance = pyo.Constraint(model.N, rule=active_power_balance_rule)

    def reactive_power_balance_rule(model, n):
        # Ensures reactive power balance at each bus
        return (sum(model.Ql[i, j] for (i, j) in model.L if j == n) - 
               sum(model.Ql[i, j] + model.X[i, j] * model.Isqr[i, j] for (i, j) in model.L if i == n) + 
               model.Qs[n] - 
               model.Qload[n] + 
               (model.Qpv[n] if n in model.G else 0) == 0)
    model.reactive_power_balance = pyo.Constraint(model.N, rule=reactive_power_balance_rule)

    def current_rule(model, i, j):
        # Defines the relationship between current, power, and voltage
        return model.Isqr[i, j] * model.Vsqr[j] == (model.Pl[i, j]**2 + model.Ql[i, j]**2)
    model.current = pyo.Constraint(model.L, rule=current_rule)

    def voltage_drop_rule(model, i, j):
        # Voltage drop calculation based on line impedance and power flow
        return (model.Vsqr[i] - 2 * (model.R[i, j] * model.Pl[i, j] + model.X[i, j] * model.Ql[i, j]) - 
               (model.R[i, j]**2 + model.X[i, j]**2) * model.Isqr[i, j] == model.Vsqr[j])
    model.voltage_drop = pyo.Constraint(model.L, rule=voltage_drop_rule)

    def ess_energy_update_rule(model, k):
        return model.E_next[k] == model.E_init[k] + model.eta_ch[k] * model.Pesc[k] - (1 / model.eta_dis[k]) * model.Pesd[k]
    model.ess_energy_update = pyo.Constraint(model.K, rule=ess_energy_update_rule)

    # Solver
    solver = SolverFactory('gurobi')
    result = solver.solve(model, tee=False)

    if result.solver.status != pyo.SolverStatus.ok:
        raise Exception('Solver failed to find a solution')

    # Extract and return the solution
    voltages = {n: np.sqrt(model.Vsqr[n].value) for n in model.N}
    currents = {(i, j): np.sqrt(model.Isqr[i, j].value) for (i, j) in model.L}
    power_flows = {(i, j): (model.Pl[i, j].value, model.Ql[i, j].value) for (i, j) in model.L}
    next_ess_energy = {k: model.E_next[k].value for k in model.K}

    return {'Voltages': voltages, 'Currents': currents, 'Power Flows': power_flows, 'Next ESS Energy': next_ess_energy}

def power_flow_solver_simplified(network_data, P_net, Q_net):

    # Define the model
    model = pyo.ConcreteModel()

    # Define Sets
    model.N = pyo.Set(initialize=network_data['bus_numbers'])  # Buses
    model.L = pyo.Set(initialize=network_data['line_connections'])  # Lines

    # Define Parameters
    model.R = pyo.Param(model.L, initialize=network_data['line_resistances'])  # Resistance of each line
    model.X = pyo.Param(model.L, initialize=network_data['line_reactances'])  # Reactance of each line
    model.Pnet = pyo.Param(model.N, initialize=P_net)  # Net active power at each bus
    model.Qnet = pyo.Param(model.N, initialize=Q_net)  # Net reactive power at each bus

    # Define Variables
    model.Vsqr = pyo.Var(model.N, within=pyo.NonNegativeReals)  # Voltage squared at each bus
    model.Pl = pyo.Var(model.L, within=pyo.Reals)  # Active power flow on each line
    model.Ql = pyo.Var(model.L, within=pyo.Reals)  # Reactive power flow on each line
    model.Isqr = pyo.Var(model.L, within=pyo.NonNegativeReals)  # Current squared on each line
    model.Ps = pyo.Var(model.N, within=pyo.Reals)  # Active power generation at each bus
    model.Qs = pyo.Var(model.N, within=pyo.Reals)  # Reactive power generation at each bus

    # Slack Bus Identification and Configuration
    for n in model.N:
        if network_data['bus_types'][n] == 1:  # Check if the bus is a slack bus
            model.Vsqr[n].fix(1)  # Fix voltage at the slack bus to 1 per unit
        else:
            model.Ps[n].fix(0)  # No generation at non-slack buses
            model.Qs[n].fix(0)  # No generation at non-slack buses

    # Constraints
    def active_power_balance_rule(model, n):
        # Active power balance at bus n
        return (sum(model.Pl[i, j] for (i, j) in model.L if j == n) - 
                sum(model.Pl[i, j] + model.R[i, j] * model.Isqr[i, j] for (i, j) in model.L if i == n) + 
                model.Ps[n] - 
                model.Pnet[n] == 0)
    model.active_power_balance = pyo.Constraint(model.N, rule=active_power_balance_rule)

    def reactive_power_balance_rule(model, n):
        # Reactive power balance at bus n
        return (sum(model.Ql[i, j] for (i, j) in model.L if j == n) - 
                sum(model.Ql[i, j] + model.X[i, j] * model.Isqr[i, j] for (i, j) in model.L if i == n) + 
                model.Qs[n] - 
                model.Qnet[n] == 0)
    model.reactive_power_balance = pyo.Constraint(model.N, rule=reactive_power_balance_rule)

    def current_rule(model, i, j):
        # Current calculation on line (i, j)
        return model.Isqr[i, j] * model.Vsqr[j] == (model.Pl[i, j] ** 2 + model.Ql[i, j] ** 2)
    model.current = pyo.Constraint(model.L, rule=current_rule)

    def voltage_drop_rule(model, i, j):
        # Voltage drop across line (i, j)
        return (model.Vsqr[i] - 2 * (model.R[i, j] * model.Pl[i, j] + model.X[i, j] * model.Ql[i, j]) - 
                (model.R[i, j] ** 2 + model.X[i, j] ** 2) * model.Isqr[i, j] == model.Vsqr[j])
    model.voltage_drop = pyo.Constraint(model.L, rule=voltage_drop_rule)

    # Solver
    solver = SolverFactory('gurobi')
    result = solver.solve(model, tee=False)

    if result.solver.status != pyo.SolverStatus.ok:
        raise Exception('Solver failed to find a solution')

    # Extract and return the solution
    voltages = {n: np.sqrt(model.Vsqr[n].value) for n in model.N}
    currents = {(i, j): np.sqrt(model.Isqr[i, j].value) for (i, j) in model.L}
    power_flows = {(i, j): (model.Pl[i, j].value, model.Ql[i, j].value) for (i, j) in model.L}

    return {'Voltages': voltages, 'Currents': currents, 'Power Flows': power_flows}