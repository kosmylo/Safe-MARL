import torch as th
import torch.nn as nn
import numpy as np
from utils.util import select_action
from madrl.models.model import Model
from madrl.critics.mlp_critic import MLPCritic
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import joblib
import logging

train_logger = logging.getLogger('TrainLogger')

class SAFEMADDPG(Model):
    def __init__(self, args, env, target_net=None):
        super(SAFEMADDPG, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net is not None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

        self.env = env
        
        # Load the pre-trained voltage predictor (linear model)
        self.voltage_predictor = joblib.load("./linear_multioutput_regressor.pkl")
        
        # Safety parameters (voltage limits)
        self.V_min = args.v_min  # Lower voltage limit
        self.V_max = args.v_max  # Upper voltage limit

        # Metrics for logging
        self.solver_interventions = 0
        self.solver_infeasible = 0
    
    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + self.n_
        else:
            input_shape = (self.obs_dim + self.act_dim) * self.n_
        output_shape = 1
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList([MLPCritic(input_shape, output_shape, self.args)])
        else:
            self.value_dicts = nn.ModuleList([MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_)])

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act):
        batch_size = obs.size(0)
        obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1)
        obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1)
        
        agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        if self.args.agent_id:
            obs_reshape = th.cat((obs_reshape, agent_ids), dim=-1)
        
        act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1)
        act_mask_others = agent_ids.unsqueeze(-1)
        act_mask_i = 1. - act_mask_others
        act_i = act_repeat * act_mask_others
        act_others = act_repeat * act_mask_i
        act_repeat = act_others.detach() + act_i
        
        if self.args.shared_params:
            obs_reshape = obs_reshape.contiguous().view(batch_size * self.n_, -1)
            act_reshape = act_repeat.contiguous().view(batch_size * self.n_, -1)
        else:
            obs_reshape = obs_reshape.contiguous().view(batch_size, self.n_, -1)
            act_reshape = act_repeat.contiguous().view(batch_size, self.n_, -1)
        
        inputs = th.cat((obs_reshape, act_reshape), dim=-1)
        
        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values, _ = agent_value(inputs, None)
            values = values.contiguous().view(batch_size, self.n_, 1)
        else:
            values = []
            for i, agent_value in enumerate(self.value_dicts):
                value, _ = agent_value(inputs[:, i, :], None)
                values.append(value)
            values = th.stack(values, dim=1)
        
        return values
    
    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            actions, log_prob_a = select_action(self.args, means, status=status, exploration=exploration, info={'log_std': log_stds})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        
        # Apply safety layer optimization
        adjusted_actions = self.safety_layer_optimization(restore_actions)

        # Convert adjusted actions from numpy array to PyTorch tensor
        adjusted_actions = th.tensor(adjusted_actions, dtype=th.float32).to(self.device)

        return adjusted_actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)

        # Get actions from the current policy
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)

        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)

        values_pol = self.value(state, actions_pol).contiguous().view(-1, self.n_)
        values = self.value(state, actions).contiguous().view(-1, self.n_)
        next_values = self.target_net.value(next_state, next_actions.detach()).contiguous().view(-1, self.n_)
        returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach()

        deltas = returns - values
        advantages = values_pol

        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)

        policy_loss = -advantages.mean()
        value_loss = deltas.pow(2).mean()

        return policy_loss, value_loss, action_out
    
    def parse_actions(self, actions):
        num_buildings = len(self.env.base_powergrid['buildings'])
        
        percentage_reduction = {}
        ess_charging = {}
        ess_discharging = {}
        q_pv = {}

        for i in range(num_buildings):
            building = self.env.base_powergrid['buildings'][i]

            # Convert actions to NumPy if they are tensors
            percentage_reduction_value = actions[0, i, 0].detach().numpy() if isinstance(actions[0, i, 0], th.Tensor) else actions[0, i, 0]
            ess_charging_value = actions[0, i, 1].detach().numpy() if isinstance(actions[0, i, 1], th.Tensor) else actions[0, i, 1]
            ess_discharging_value = actions[0, i, 2].detach().numpy() if isinstance(actions[0, i, 2], th.Tensor) else actions[0, i, 2]
            q_pv_value = actions[0, i, 3].detach().numpy() if isinstance(actions[0, i, 3], th.Tensor) else actions[0, i, 3]

            # Assign the converted or raw values
            percentage_reduction[building] = self.env.args.max_power_reduction * percentage_reduction_value
            ess_charging[building] = self.env.args.p_ch_max * ess_charging_value
            ess_discharging[building] = self.env.args.p_dis_max * ess_discharging_value
            q_pv[building] = self.env._scale_and_clip_q_pv(q_pv_value, self.env.current_pv_power[building])

        # Clip the power reduction percentages to be within the allowed range
        percentage_reduction = self.env.clip_percentage_reduction(percentage_reduction)
        
        # Adjust ESS charging/discharging to prevent simultaneous actions
        ess_charging, ess_discharging = self.env.adjust_ess_actions(ess_charging, ess_discharging)

        for k in ess_charging:
            ess_charging[k], ess_discharging[k] = self.env._clip_power_charging_discharging(ess_charging[k], ess_discharging[k], self.env.current_ess_energy[k])

        return percentage_reduction, ess_charging, ess_discharging, q_pv

    def safety_layer_optimization(self, proposed_actions):
        
        # Parse the proposed actions
        percentage_reduction, ess_charging, ess_discharging, q_pv = self.parse_actions(proposed_actions)

        # Extract the coefficients for each output (bus) from the voltage predictor
        W_P = np.array([est.coef_[:len(self.env.base_powergrid['bus_numbers'])] for est in self.voltage_predictor.estimators_])
        W_Q = np.array([est.coef_[len(self.env.base_powergrid['bus_numbers']):] for est in self.voltage_predictor.estimators_])
        b = np.array([est.intercept_ for est in self.voltage_predictor.estimators_])

        # Create the optimization model
        model = pyo.ConcreteModel()

        # Define Sets
        model.N = pyo.Set(initialize=self.env.base_powergrid['bus_numbers']) # Buses
        model.B = pyo.Set(initialize=self.env.base_powergrid['buildings'])  # Buildings with flexibility
        model.G = pyo.Set(initialize=self.env.base_powergrid['PVs_at_buildings'])  # PVs
        model.K = pyo.Set(initialize=self.env.base_powergrid['ESSs_at_buildings'])  # ESSs

        # Define decision variables for the buses with buildings
        model.percentage_reduction = pyo.Var(model.B, domain=pyo.NonNegativeReals)
        model.ess_charging = pyo.Var(model.K, domain=pyo.NonNegativeReals)
        model.ess_discharging = pyo.Var(model.K, domain=pyo.NonNegativeReals)
        model.q_pv = pyo.Var(model.G, domain=pyo.Reals)

        # Define P_net and Q_net as decision variables for all buses
        model.P_net = pyo.Var(model.N, domain=pyo.Reals)
        model.Q_net = pyo.Var(model.N, domain=pyo.Reals)

        # Add slack variables to relax the voltage constraint and avoid infeasibility
        model.slack_lower = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        model.slack_upper = pyo.Var(model.N, domain=pyo.NonNegativeReals)

        # Define penalty coefficients for slack variables
        penalty_lower = 1000  
        penalty_upper = 1000

        # Objective: minimize deviation from proposed actions for buses with buildings
        def objective_rule(model):
            deviation = sum(
                (model.percentage_reduction[b] - percentage_reduction[b])**2 for b in model.B
            ) + sum(
                (model.ess_charging[k] - ess_charging[k])**2 for k in model.K
            ) + sum(
                (model.ess_discharging[k] - ess_discharging[k])**2 for k in model.K
            ) + sum(
                (model.q_pv[g] - q_pv[g])**2 for g in model.G
            ) 
            
            slack_penalty = sum(
                penalty_lower * model.slack_lower[n] + penalty_upper * model.slack_upper[n] for n in model.N
            )

            return deviation + slack_penalty

        # Create the objective using the rule
        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Rule for P_net constraint
        def P_net_rule(model, bus):
            # Initialize with current active demand
            P_net = self.env.current_active_demand[bus]

            # If the bus has a building, apply percentage reduction and ESS charging/discharging
            if bus in model.B:
                P_net *= (1 - model.percentage_reduction[bus])

            if bus in model.K:
                P_net += model.ess_charging[bus] - model.ess_discharging[bus]

            return model.P_net[bus] == P_net

        # Rule for Q_net constraint
        def Q_net_rule(model, bus):
            # Initialize with current reactive demand
            Q_net = self.env.current_reactive_demand[bus]

            # If the bus has a PV, apply PV reactive power adjustment
            if bus in model.G:
                Q_net += model.q_pv[bus]

            return model.Q_net[bus] == Q_net

        # Add constraints for P_net and Q_net for all buses
        model.P_net_constraints = pyo.Constraint(model.N, rule=P_net_rule)
        model.Q_net_constraints = pyo.Constraint(model.N, rule=Q_net_rule)

        # Define voltage prediction lower bound constraint as a rule
        def voltage_constraint_lower_rule(model, bus, W_P, W_Q, b):
            bus_idx = self.env.base_powergrid['bus_numbers'].index(bus)
            V_pred = sum(W_P[bus_idx] * model.P_net[bus] + W_Q[bus_idx] * model.Q_net[bus]) + b[bus_idx]
            return self.V_min - model.slack_lower[bus]<= V_pred

        # Define voltage prediction upper bound constraint as a rule
        def voltage_constraint_upper_rule(model, bus, W_P, W_Q, b):
            bus_idx = self.env.base_powergrid['bus_numbers'].index(bus)
            V_pred = sum(W_P[bus_idx] * model.P_net[bus] + W_Q[bus_idx] * model.Q_net[bus]) + b[bus_idx]
            return V_pred <= self.V_max + model.slack_upper[bus]

        # Add voltage constraints for all buses
        model.voltage_constraints_lower = pyo.Constraint(model.N, rule=lambda model, bus: voltage_constraint_lower_rule(model, bus, W_P, W_Q, b))
        model.voltage_constraints_upper = pyo.Constraint(model.N, rule=lambda model, bus: voltage_constraint_upper_rule(model, bus, W_P, W_Q, b))

        # Solve the optimization problem
        solver = SolverFactory('gurobi')
        results = solver.solve(model, tee=False)

        # Handle infeasible cases
        if (results.solver.status != pyo.SolverStatus.ok) or (results.solver.termination_condition != pyo.TerminationCondition.optimal):
            print(results.solver.terminate_condition)
            self.solver_infeasible += 1
            train_logger.info(f"Optimization problem infeasible or unbounded. Infeasibility count: {self.solver_infeasible}")
            return proposed_actions  # Return original actions if infeasible

        # Extract optimized actions for buses with buildings
        optimized_percentage_reduction = np.array([model.percentage_reduction[bus].value for bus in model.B])
        optimized_ess_charging = np.array([model.ess_charging[bus].value for bus in model.K])
        optimized_ess_discharging = np.array([model.ess_discharging[bus].value for bus in model.K])
        optimized_q_pv = np.array([model.q_pv[bus].value for bus in model.G])

        # Adjusted actions as a single array
        adjusted_actions = np.concatenate([optimized_percentage_reduction, optimized_ess_charging, optimized_ess_discharging, optimized_q_pv])

        return adjusted_actions
