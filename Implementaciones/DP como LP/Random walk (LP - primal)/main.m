% Analysis Random Walk problem: DP as an LP
clear all, close all, clc
random_walk_set_up;

S=R_W.N_states; % Para state value function
A=R_W.N_actions;
% n=R_W.N_states*R_W.N_actions; % Para state-action value function

cvx_begin
    variable v(S)
    minimize( sum(v) )
    subject to
%         for s=1:S
%             for a=1:A
%                 v(s) >= R((s-1)*A + a) + gamma*P((s-1)*A + a,:)*v
%             end
%         end
        R + gamma*P*v <= marg'*v
%         pi_a1*R + gamma*pi_a1*P*v <= v
%         pi_a2*R + gamma*pi_a2*P*v <= v
%         R + gamma*P*pi_a2*v <= v
%         R + gamma*P*pi_a1*v <= v
cvx_end

v
q = R+gamma*P*v;
reshape(q', [2,7])';
final_policy = (q-R)./sum(duplicar*diag(gamma*q'*P),2);