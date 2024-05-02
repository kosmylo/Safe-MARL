import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from constants import T, V_MIN, V_MAX, FLEX_PRICE, PV_COST, ESS_COST, DISCOMFORT_COEFF, ETA_CH, ETA_DIS, LOAD_PROFILE, PV_CAPACITY, PV_PROFILE, MAX_POWER_REDUCTION_PERCENT, E_MIN, E_MAX, P_CH_MAX, P_DIS_MAX
import logging

logger = logging.getLogger(__name__)

def opf_model(network_data):

    # Define the model
    model = pyo.ConcreteModel()

    # Define Sets
    model.T = pyo.RangeSet(1, T)  # Time periods
    model.N = pyo.Set(initialize=network_data['bus_numbers']) # Buses
    model.L = pyo.Set(initialize=network_data['line_connections'])  # Lines
    model.B = pyo.Set(initialize=network_data['buildings'])  # Buildings with flexibility
    model.G = pyo.Set(initialize=network_data['PVs_at_buildings'])  # PVs
    model.K = pyo.Set(initialize=network_data['ESSs_at_buildings'])  # ESSs

    # Define Parameters
    model.R = pyo.Param(model.L, initialize=network_data['line_resistances']) # Resistance of each line
    model.X = pyo.Param(model.L, initialize=network_data['line_reactances']) # Reactance of each line
    model.Vmin = pyo.Param(initialize=V_MIN)  # Minimum voltage at each bus
    model.Vmax = pyo.Param(initialize=V_MAX)  # Maximum voltage at each bus
    model.Imax = pyo.Param(model.L, initialize=network_data['max_line_currents'])  # Max capacity of each line
    model.lambda_flex = pyo.Param(model.T, initialize=FLEX_PRICE)  # Price of flexibility per time period
    model.cost_pv = pyo.Param(model.G, initialize=PV_COST)  # Cost of reactive power control of each PV
    model.cost_ess = pyo.Param(model.K, initialize=ESS_COST)  # Cost of operating each ESS
    model.discomfort = pyo.Param(model.B, initialize=DISCOMFORT_COEFF)  # Discomfort coefficient
    model.eta_ch = pyo.Param(model.K, initialize=ETA_CH)
    model.eta_dis = pyo.Param(model.K, initialize=ETA_DIS)
    model.Pload = pyo.Param(model.N, model.T, initialize=lambda model, n, t: network_data['active_power_demand'][n] * LOAD_PROFILE[t-1])
    model.Qload = pyo.Param(model.N, model.T, initialize=lambda model, n, t: network_data['reactive_power_demand'][n] * LOAD_PROFILE[t-1])
    model.Ppv = pyo.Param(model.G, model.T, initialize=lambda model, g, t: PV_CAPACITY * PV_PROFILE[t-1])
    model.Pred_max = pyo.Param(model.B, model.T, initialize=lambda model, b, t: network_data['active_power_demand'][b] * LOAD_PROFILE[t-1] * MAX_POWER_REDUCTION_PERCENT)
    model.Emin = pyo.Param(initialize=E_MIN)
    model.Emax = pyo.Param(initialize=E_MAX)
    model.Pch_max = pyo.Param(initialize=P_CH_MAX)
    model.Pdis_max = pyo.Param(initialize=P_DIS_MAX)

    # Define Variables
    model.Pred = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals, bounds=lambda model, b, t: (0, model.Pred_max[b, t]))  # Power reduction by building
    # model.Qpv = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)  # Reactive power provided by PVs
    model.Pesc = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals, bounds=(0, model.Pch_max))  # Power charged to ESS
    model.Pesd = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals, bounds=(0, model.Pdis_max)) # Power discharged from ESS
    # model.V = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)  # Voltage at each bus
    model.E = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals, bounds=(model.Emin, model.Emax)) # Energy of the ESS
    model.Pl = pyo.Var(model.L, model.T, within=pyo.Reals)  # Active power flow on each line
    model.Ql = pyo.Var(model.L, model.T, within=pyo.Reals)  # Reactive power flow on each line
    model.I = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)  # Current flow on each line

    # Define Variables with conditional initialization
    model.V = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, initialize=lambda model, n, t: 1 if network_data['bus_types'][n] == 1 else 1)
    for n in model.N:
        if network_data['bus_types'][n] == 1:
            model.V[n, :].fixed = True  # Fix voltage at substations

    model.Ps = pyo.Var(model.N, model.T, within=pyo.Reals, initialize=lambda model, n, t: 0 if network_data['bus_types'][n] == 0 else 0)
    for n in model.N:
        if network_data['bus_types'][n] == 0:
            model.Ps[n, :].fixed = True  # No active power generation at non-substation nodes

    model.Qs = pyo.Var(model.N, model.T, within=pyo.Reals, initialize=lambda model, n, t: 0 if network_data['bus_types'][n] == 0 else 0)
    for n in model.N:
        if network_data['bus_types'][n] == 0:
            model.Qs[n, :].fixed = True  # No reactive power generation at non-substation nodes

    # Define Objective Function
    def objective_rule(model):
        return sum(model.lambda_flex[t] * model.Pred[b, t] - model.cost_ess[k] * (model.Pesc[k, t] + model.Pesd[k, t]) - model.discomfort[b] * model.Pred[b, t] for b in model.B for k in model.K for t in model.T)
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)    

    # Define Constraints

    def active_power_flow_rule(model, n, t):
        # Sum of outgoing power - Sum of incoming power + generation - load = 0 at each bus
        return (sum(model.Pl[i, j, t] for (i, j) in model.L if j == n) -
                sum(model.Pl[i, j, t] + model.R[i, j] * model.I[i, j, t] for (i, j) in model.L if i == n) +
                sum(model.Pesd[k, t] for k in model.K if k == n) -  
                sum(model.Pesc[k, t] for k in model.K if k == n) -  
                sum(model.Pred[b, t] for b in model.B if b == n) -  
                sum(model.Ppv[g, t] for g in model.G if g == n) +
                model.Ps[n, t] -
                model.Pload[n, t] == 0)
    model.active_power_flow = pyo.Constraint(model.N, model.T, rule=active_power_flow_rule)

    def reactive_power_flow_rule(model, n, t):
        return (sum(model.Ql[i, j, t] for (i, j) in model.L if j == n) -
                sum(model.Ql[i, j, t] + model.X[i, j] * model.I[i, j, t] for (i, j) in model.L if i == n) +
                #  + sum(model.Qpv[g, t] for g in model.G if g == n) -
                model.Qs[n, t] - 
                model.Qload[n, t] == 0)
    model.reactive_power_flow = pyo.Constraint(model.N, model.T, rule=reactive_power_flow_rule)

    def voltage_drop_rule(model, i, j, t):
        return model.V[i, t] - 2 * (model.R[i, j] * model.Pl[i, j, t] + model.X[i, j] * model.Ql[i, j, t]) - \
            (model.R[i, j]**2 + model.X[i, j]**2) * model.I[i, j, t] == model.V[j, t]
    model.voltage_drop = pyo.Constraint(model.L, model.T, rule=voltage_drop_rule)

    def define_current_rule(model, i, j, t):
        return model.I[i, j, t] * model.V[j, t] == (model.Pl[i, j, t]**2 + model.Ql[i, j, t]**2)
    model.define_current = pyo.Constraint(model.L, model.T, rule=define_current_rule)

    def current_limit_rule(model, i, j, t):
        return model.I[i, j, t] <= model.Imax[i, j]**2
    model.current_limit = pyo.Constraint(model.L, model.T, rule=current_limit_rule)

    def voltage_limit_rule(model, n, t):
        return (model.Vmin**2, model.V[n, t], model.Vmax**2)
    model.voltage_limit = pyo.Constraint(model.N, model.T, rule=voltage_limit_rule)

    def ess_energy_balance_rule(model, k, t):
        if t == 1:
            # Assume initial energy content set to E_min 
            return model.E[k, t] ==  model.Emin
        else:
            # Energy at time t based on the energy at time t-1 plus net energy changes
            return (model.E[k, t] == model.E[k, t-1] + 
                    (model.eta_ch[k] * model.Pesc[k, t-1] - 
                     (1 / model.eta_dis[k]) * model.Pesd[k, t-1]))
    model.ess_energy_balance = pyo.Constraint(model.K, model.T, rule=ess_energy_balance_rule)

    def no_simultaneous_charge_discharge_rule(model, k, t):
        return model.Pesc[k, t] * model.Pesd[k, t] == 0
    model.no_simultaneous_charge_discharge = pyo.Constraint(model.K, model.T, rule=no_simultaneous_charge_discharge_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    result = solver.solve(model, tee=True)
    
    if result.solver.status != pyo.SolverStatus.ok:
        logger.error(f'Solver failed to find a solution')
        raise Exception('Solver failed to find a solution')
    
        # Extract and store the solution
    solution = {
        'Power Reduction': {},
        # 'PV Reactive Power': {},
        'ESS Charging': {},
        'ESS Discharging': {},
        'Voltage': {},
        'Active Power Flow': {},
        'Reactive Power Flow': {},
        'Current': {},
        'ESS Energy': {}
    }

    # Extract results for each time period and component
    for t in model.T:
        solution['Power Reduction'][t] = {b: model.Pred[b, t].value for b in model.B}
        # solution['PV Reactive Power'][t] = {g: model.Qpv[g, t].value for g in model.G}
        solution['ESS Charging'][t] = {k: model.Pesc[k, t].value for k in model.K}
        solution['ESS Discharging'][t] = {k: model.Pesd[k, t].value for k in model.K}
        solution['Voltage'][t] = {n: model.V[n, t].value for n in model.N}
        solution['Active Power Flow'][t] = {(i, j): model.Pl[i, j, t].value for (i, j) in model.L}
        solution['Reactive Power Flow'][t] = {(i, j): model.Ql[i, j, t].value for (i, j) in model.L}
        solution['Current'][t] = {(i, j): model.I[i, j, t].value for (i, j) in model.L}
        solution['ESS Energy'][t] = {k: model.E[k, t].value for k in model.K}

    return solution

