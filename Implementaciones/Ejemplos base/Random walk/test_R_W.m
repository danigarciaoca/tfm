% Analysis Random Walk problem
clear all
random_walk_set_up;
N_steps_value_ite=4000;
% Optimum value
v_opt=inv(eye(R_W.N_states)-R_W.gamma*R_W.pi_opt*R_W.P)*R_W.pi_opt*R_W.R;
q_opt=inv(eye(R_W.N_states*R_W.N_actions)-R_W.gamma*R_W.P*R_W.pi_opt)*R_W.R;

% Solution of the Bellman linear equations
% Solution of the V/Q equations
[v_rp q_rp]=bellman_linear(R_W);
[v_pe_bell q_pe_bell]=policy_evaluation(R_W,N_steps_value_ite);

% We plot the results
aux=repmat(v_rp,[1 N_steps_value_ite])-v_pe_bell;
plot(1:N_steps_value_ite,10*log10(diag(aux'*aux)),'b','LIneWidth',5),grid
hold
aux=repmat(q_rp,[1 N_steps_value_ite])-q_pe_bell;
plot(1:N_steps_value_ite,10*log10(diag(aux'*aux)),'r','LIneWidth',5)
xlabel('Steps')
ylabel('Error')
title('Value / Q iteration performance'),hold off

% Value iteration
[v_vi q_vi] = value_iteration(R_W,N_steps_value_ite);

% Policy iteration
%cliff_set_up
[v_pi q_pi] = policy_improvement(R_W,N_steps_value_ite);
[v_vi v_pi(:,N_steps_value_ite)]
aux=repmat(v_vi,[1 N_steps_value_ite])-v_pi;
plot(1:N_steps_value_ite,10*log10(diag(aux'*aux)))
xlabel('Steps')
ylabel('Policy improvement')

% TD V function. Random policy
[v_td] = TD_R_W(R_W,N_steps_value_ite);
[v_rp v_td(:,N_steps_value_ite)]

% QL algorithm
R_W.alpha=.1;
[q_ql] = QL_R_W(R_W,N_steps_value_ite);
aux=repmat(q_opt,[1 N_steps_value_ite])-q_ql;
plot(1:N_steps_value_ite,10*log10(sum(aux.^2,1)),'b','LIneWidth',5),grid

[q_ql(1:2:end,end) v_opt]


% SARSA algorithm
R_W.alpha=.1;
[q_sarsa] = SARSA_R_W(R_W,N_steps_value_ite);
aux=repmat(q_opt,[1 N_steps_value_ite])-q_sarsa;
plot(1:N_steps_value_ite,10*log10(sum(aux.^2,1)),'b','LIneWidth',5),grid

[q_sarsa(1:2:end,end) v_opt]


