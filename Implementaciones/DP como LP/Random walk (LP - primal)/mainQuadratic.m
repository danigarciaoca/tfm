% Analysis Random Walk problem: DP as an LP
clear all, close all, clc
random_walk_set_up;

S=R_W.N_states; % Para state value function
A=R_W.N_actions;
% n=R_W.N_states*R_W.N_actions; % Para state-action value function

cvx_begin
    variable v(S)
    minimize( mu'*v )
    subject to
        exp(R + gamma*P*v - marg'*v) - 1 <= zeros(S*A,1)
cvx_end

v
q = R+gamma*P*v;
final_policy = (q-R)./sum(duplicar*diag(gamma*q'*P),2); % CACA