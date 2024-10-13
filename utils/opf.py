import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from math import acos, tan
import yaml
import logging

logger = logging.getLogger(__name__)

# load env args
with open("./madrl/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]

def opf_model(network_data, flex_price, active_power_demand, reactive_power_demand, pv_active_power, initial_ess_energy):

    # Define the model
    model = pyo.ConcreteModel()

    # Define Sets
    model.T = pyo.RangeSet(1, env_config_dict['episode_limit'])  # Time periods
    model.N = pyo.Set(initialize=network_data['bus_numbers']) # Buses
    model.L = pyo.Set(initialize=network_data['line_connections'])  # Lines
    model.B = pyo.Set(initialize=network_data['buildings'])  # Buildings with flexibility
    model.G = pyo.Set(initialize=network_data['PVs_at_buildings'])  # PVs
    model.K = pyo.Set(initialize=network_data['ESSs_at_buildings'])  # ESSs

    # Calculate time step duration
    total_hours = 24  # Assuming a 24-hour period
    time_step_duration = total_hours / env_config_dict['episode_limit']

    # Define Parameters
    model.delta_t = pyo.Param(initialize=time_step_duration)
    model.R = pyo.Param(model.L, initialize=network_data['line_resistances']) # Resistance of each line
    model.X = pyo.Param(model.L, initialize=network_data['line_reactances']) # Reactance of each line
    model.Vmin = pyo.Param(initialize=env_config_dict['v_min'])  # Minimum voltage at each bus
    model.Vmax = pyo.Param(initialize=env_config_dict['v_max'])  # Maximum voltage at each bus
    model.Imax = pyo.Param(model.L, initialize=network_data['max_line_currents'])  # Max capacity of each line
    model.lambda_flex = pyo.Param(model.T, initialize=flex_price)  # Price of flexibility per time period
    model.cost_pv = pyo.Param(model.G, initialize=env_config_dict['pv_cost'])  # Cost of reactive power control of each PV
    model.cost_ess = pyo.Param(model.K, initialize=env_config_dict['ess_cost'])  # Cost of operating each ESS
    model.discomfort = pyo.Param(model.B, initialize=env_config_dict['discomfort_coeff'])  # Discomfort coefficient
    model.eta_ch = pyo.Param(model.K, initialize=env_config_dict['eta_ch'])
    model.eta_dis = pyo.Param(model.K, initialize=env_config_dict['eta_dis'])
    model.Pload = pyo.Param(model.N, model.T, initialize=lambda model, n, t: active_power_demand[n][t-1])
    model.Qload = pyo.Param(model.N, model.T, initialize=lambda model, n, t: reactive_power_demand[n][t-1])
    model.Ppv = pyo.Param(model.G, model.T, initialize=lambda model, g, t: pv_active_power[g][t-1])
    model.Pred_max = pyo.Param(model.B, model.T, initialize=lambda model, b, t: active_power_demand[b][t-1] * env_config_dict['max_power_reduction'])
    model.Emin = pyo.Param(initialize=env_config_dict['e_min'])  # Minimum energy of the ESS
    model.Emax = pyo.Param(initialize=env_config_dict['e_max']) # Maximum energy of the ESS
    model.Pch_max = pyo.Param(initialize=env_config_dict['p_ch_max']) # Maximum charging power of the ESS
    model.Pdis_max = pyo.Param(initialize=env_config_dict['p_dis_max']) # Maximum discharging power of the ESS
    model.E_init = pyo.Param(model.K, initialize=initial_ess_energy)  # Initial ESS energy state

    # Define Variables
    model.Pred = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals, bounds=lambda model, b, t: (0, model.Pred_max[b, t]))  # Power reduction by building
    model.Qpv = pyo.Var(model.G, model.T, within=pyo.Reals)  # Reactive power provided by PVs
    model.Pesc = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals, bounds=(0, model.Pch_max))  # Power charged to ESS
    model.Pesd = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals, bounds=(0, model.Pdis_max)) # Power discharged from ESS
    model.E = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals, bounds=(model.Emin, model.Emax)) # Energy of the ESS
    model.Pl = pyo.Var(model.L, model.T, within=pyo.Reals)  # Active power flow on each line
    model.Ql = pyo.Var(model.L, model.T, within=pyo.Reals)  # Reactive power flow on each line
    model.Isqr = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)  # Current flow on each line
    model.charging_indicator = pyo.Var(model.K, model.T, within=pyo.Binary) # Binary variable that shows if the ESS is charging
    model.Vsqr = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, initialize=1)
    model.Ps = pyo.Var(model.N, model.T, within=pyo.Reals, initialize=0)
    model.Qs = pyo.Var(model.N, model.T, within=pyo.Reals, initialize=0)

    # Fix Variables at Substations and Non-Substation Nodes
    for n in model.N:
        for t in model.T:
            if network_data['bus_types'][n] == 1:  # Substation bus
                model.Vsqr[n, t].fix(1)
            else:  # Non-substation bus
                model.Ps[n, t].fix(0)
                model.Qs[n, t].fix(0)

    # Define Objective Function
    def objective_rule(model):
        return sum(
            model.delta_t * (
                sum(model.lambda_flex[t] * model.Pred[b, t] for b in model.B)
                - sum(model.cost_pv[g] * model.Qpv[g, t] for g in model.G)
                - sum(model.cost_ess[k] * (model.Pesc[k, t] + model.Pesd[k, t]) for k in model.K)
                - sum(model.R[i, j] * model.Isqr[i, j, t] for (i, j) in model.L)
                - sum(model.discomfort[b] * model.Pred[b, t]**2 for b in model.B)
            )
            for t in model.T
        )
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Define Constraints
    def active_power_flow_rule(model, n, t):
        # Sum of outgoing power - Sum of incoming power + generation - load = 0 at each bus
        return (sum(model.Pl[i, j, t] for (i, j) in model.L if j == n) -
                sum(model.Pl[i, j, t] + model.R[i, j] * model.Isqr[i, j, t] for (i, j) in model.L if i == n) +
                sum(model.Pesd[k, t] for k in model.K if k == n) -  
                sum(model.Pesc[k, t] for k in model.K if k == n) +  
                sum(model.Ppv[g, t] for g in model.G if g == n) +
                model.Ps[n, t] -
                model.Pload[n, t] +
                sum(model.Pred[b, t] for b in model.B if b == n)  == 0)
    model.active_power_flow = pyo.Constraint(model.N, model.T, rule=active_power_flow_rule)

    def reactive_power_flow_rule(model, n, t):
        return (sum(model.Ql[i, j, t] for (i, j) in model.L if j == n) -
                sum(model.Ql[i, j, t] + model.X[i, j] * model.Isqr[i, j, t] for (i, j) in model.L if i == n) +
                sum(model.Qpv[g, t] for g in model.G if g == n) +
                model.Qs[n, t] - 
                model.Qload[n, t] == 0)
    model.reactive_power_flow = pyo.Constraint(model.N, model.T, rule=reactive_power_flow_rule)

    def Qpv_constraint_rule(model, g, t):
        return (-tan(acos(env_config_dict['cos_phi_max'])) * model.Ppv[g, t], model.Qpv[g, t], tan(acos(env_config_dict['cos_phi_max'])) * model.Ppv[g, t])
    model.Qpv_control = pyo.Constraint(model.G, model.T, rule=Qpv_constraint_rule)

    def voltage_drop_rule(model, i, j, t):
        return model.Vsqr[i, t] - 2 * (model.R[i, j] * model.Pl[i, j, t] + model.X[i, j] * model.Ql[i, j, t]) - \
            (model.R[i, j]**2 + model.X[i, j]**2) * model.Isqr[i, j, t] == model.Vsqr[j, t]
    model.voltage_drop = pyo.Constraint(model.L, model.T, rule=voltage_drop_rule)

    def define_current_rule(model, i, j, t):
        return model.Isqr[i, j, t] * model.Vsqr[j, t] == (model.Pl[i, j, t]**2 + model.Ql[i, j, t]**2)
    model.define_current = pyo.Constraint(model.L, model.T, rule=define_current_rule)

    def current_limit_rule(model, i, j, t):
        return model.Isqr[i, j, t] <= model.Imax[i, j]**2
    model.current_limit = pyo.Constraint(model.L, model.T, rule=current_limit_rule)

    def voltage_limit_rule(model, n, t):
        return (model.Vmin**2, model.Vsqr[n, t], model.Vmax**2)
    model.voltage_limit = pyo.Constraint(model.N, model.T, rule=voltage_limit_rule)

    def ess_energy_balance_rule(model, k, t):
        if t == 1:
            # Assume initial energy content set to initial_ess_energy
            return model.E[k, t] ==  model.E_init[k]
        else:
            # Energy at time t based on the energy at time t-1 plus net energy changes
            return (model.E[k, t] == model.E[k, t-1] + 
                    model.delta_t * (model.eta_ch[k] * model.Pesc[k, t] - 
                     (1 / model.eta_dis[k]) * model.Pesd[k, t]))
    model.ess_energy_balance = pyo.Constraint(model.K, model.T, rule=ess_energy_balance_rule)

    def no_simultaneous_charge_rule(model, k, t):
        return model.Pesc[k, t] <= model.Pch_max * model.charging_indicator[k, t]
    model.no_simultaneous_charge = pyo.Constraint(model.K, model.T, rule=no_simultaneous_charge_rule)

    def no_simultaneous_discharge_rule(model, k, t):
        return model.Pesd[k, t] <= model.Pdis_max * (1 - model.charging_indicator[k, t])
    model.no_simultaneous_discharge = pyo.Constraint(model.K, model.T, rule=no_simultaneous_discharge_rule)

    # Solve the model
    solver = SolverFactory('gurobi')
    result = solver.solve(model, tee=False)
    
    if result.solver.status != pyo.SolverStatus.ok:
        logger.error(f'Solver failed to find a solution')
        raise Exception('Solver failed to find a solution')
    
    # Extract and store the solution
    solution = {
        'Power Reduction': {},
        'PV Reactive Power': {},
        'ESS Charging': {},
        'ESS Discharging': {},
        'Voltage Squared': {},
        'Active Power Flow': {},
        'Reactive Power Flow': {},
        'Current Squared': {},
        'ESS Energy': {},
        'Charging Indicator': {},
        'Active Power Load': {},  
        'Reactive Power Load': {}, 
        'PV Active Power': {}  
    }

    # Extract results for each time period and component
    for t in model.T:
        solution['Power Reduction'][t] = {b: model.Pred[b, t].value for b in model.B}
        solution['PV Reactive Power'][t] = {g: model.Qpv[g, t].value for g in model.G}
        solution['ESS Charging'][t] = {k: model.Pesc[k, t].value for k in model.K}
        solution['ESS Discharging'][t] = {k: model.Pesd[k, t].value for k in model.K}
        solution['Voltage Squared'][t] = {n: model.Vsqr[n, t].value for n in model.N}
        solution['Active Power Flow'][t] = {(i, j): model.Pl[i, j, t].value for (i, j) in model.L}
        solution['Reactive Power Flow'][t] = {(i, j): model.Ql[i, j, t].value for (i, j) in model.L}
        solution['Current Squared'][t] = {(i, j): model.Isqr[i, j, t].value for (i, j) in model.L}
        solution['ESS Energy'][t] = {k: model.E[k, t].value for k in model.K}
        solution['Charging Indicator'][t] = {k: model.charging_indicator[k, t].value for k in model.K}
        solution['Active Power Load'][t] = {n: model.Pload[n, t] for n in model.N}
        solution['Reactive Power Load'][t] = {n: model.Qload[n, t] for n in model.N}
        solution['PV Active Power'][t] = {g: model.Ppv[g, t] for g in model.G}

    return solution

