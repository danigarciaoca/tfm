% Analysis Random Walk problem: DP as an LP
clear all, close all, clc
random_walk_set_up;

S=R_W.N_states;
A=R_W.N_actions;
PI = S*A;

cvx_begin
    variable d(PI)
    maximize( d.'*R )
    subject to
        marg*d == (1-gamma)*mu + gamma*P.'*d;
        d >= 0;
cvx_end

d
sumDoverA_aux = sum(reshape(d,[A,S]),1);
sumDoverA = sum(duplicar*diag(sumDoverA_aux),2);
policy = d./sumDoverA;
policy_by_action = reshape(policy', [A,S])' % primera columna derecha, segunda izquierda
final_policy = policy_by_action == max(policy_by_action,[],2);
table(policy_by_action)
q = R+gamma*P*v