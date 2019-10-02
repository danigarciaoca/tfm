function [v q] = policy_evaluation( problem,N_steps)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% Policy evaluation. V / Q functions
v=zeros(problem.N_states,N_steps);
q=zeros(problem.N_states*problem.N_actions,N_steps);
for k=1:N_steps-1
    v(:,k+1)=problem.pi_rp*(problem.R+problem.gamma*problem.P*v(:,k));
    q(:,k+1)=(problem.R+problem.gamma*problem.P*problem.pi_rp*q(:,k));
end
end

