import torch as th
import torch.nn as nn
import numpy as np
from utils.util import select_action
from madrl.models.model import Model
from madrl.critics.mlp_critic import MLPCritic
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory, NonNegativeReals
import joblib

class SAFEMADDPG(Model):
    def __init__(self, args, target_net=None):
        super(SAFEMADDPG, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net is not None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)
        
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
        num_buildings = len(self.base_powergrid['buildings'])
        
        percentage_reduction = {}
        ess_charging = {}
        ess_discharging = {}
        q_pv = {}

        for i in range(num_buildings):
            building = self.base_powergrid['buildings'][i]
            percentage_reduction[building] = self.args.max_power_reduction * actions[i * 4]
            ess_charging[building] = self.args.p_ch_max * actions[i * 4 + 1]
            ess_discharging[building] = self.args.p_dis_max * actions[i * 4 + 2]
            q_pv[building] = self._scale_and_clip_q_pv(actions[i * 4 + 3], self.current_pv_power[building])

        # Adjust ESS charging/discharging to prevent simultaneous actions
        ess_charging, ess_discharging = self.adjust_ess_actions(ess_charging, ess_discharging)

        return percentage_reduction, ess_charging, ess_discharging, q_pv

    def compute_net_powers(self, percentage_reduction, ess_charging, ess_discharging, q_pv):
        P_net = {}
        Q_net = {}

        for bus in self.base_powergrid['bus_numbers']:
            # Net Active Power (P_net)
            P_net[bus] = self.current_active_demand[bus]

            if bus in self.base_powergrid['buildings']:
                # Apply power reduction
                P_net[bus] *= (1 - percentage_reduction[bus])

            if bus in self.base_powergrid['ESSs_at_buildings']:
                # Apply ESS charging/discharging
                P_net[bus] += ess_charging[bus] - ess_discharging[bus]

            # Net Reactive Power (Q_net)
            Q_net[bus] = self.current_reactive_demand[bus]

            if bus in self.base_powergrid['PVs_at_buildings']:
                # Adjust reactive power from PV system
                Q_net[bus] += q_pv[bus]

        return P_net, Q_net
    
    def predict_voltage(self, P_net, Q_net):
        inputs = np.hstack([list(P_net.values()), list(Q_net.values())])
        predicted_voltages = self.voltage_predictor.predict([inputs])
        return predicted_voltages[0]

    def safety_layer_optimization(self, proposed_actions):
        # Parse the proposed actions
        percentage_reduction, ess_charging, ess_discharging, q_pv = self.parse_actions(proposed_actions)

        # Compute net active and reactive power
        P_net, Q_net = self.compute_net_powers(percentage_reduction, ess_charging, ess_discharging, q_pv)

        # Predict voltages
        predicted_voltages = self.predict_voltage(P_net, Q_net)

        model = ConcreteModel()

        # Define decision variables for the actions
        n_actions = len(proposed_actions)
        model.a = Var(range(n_actions), domain=NonNegativeReals)

        # Objective: minimize || a_t - pi(o_t) ||^2
        model.obj = Objective(expr=sum((model.a[i] - proposed_actions[i])**2 for i in range(n_actions)))

        # Voltage constraints
        model.constraints = ConstraintList()

        for n, v_hat in enumerate(predicted_voltages):
            model.constraints.add(self.V_min <= v_hat <= self.V_max)

        # Solve the optimization problem
        solver = SolverFactory('gurobi')
        results = solver.solve(model, tee=False)

        # Handle infeasible cases
        if results.solver.status == 'ok':
            adjusted_actions = np.array([model.a[i].value for i in range(n_actions)])
        else:
            self.solver_infeasible += 1
            adjusted_actions = proposed_actions  # Return original actions if infeasible

        # Count solver interventions if adjustments are made
        if np.linalg.norm(adjusted_actions - proposed_actions) > 1e-3:
            self.solver_interventions += 1

        return adjusted_actions
