function [v_opt q_opt] = value_iteration(problem,N_steps)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% Policy evaluation. V / Q functions
v=zeros(problem.N_states,N_steps);
v=problem.v_ini;
q=zeros(problem.N_states*problem.N_actions,N_steps);
q(:,1)=problem.R;
for k=1:N_steps-1
    for kk=1:problem.N_states
        v(kk,k+1)=max(q((kk-1)*problem.N_actions+1:kk*problem.N_actions,k));
    end
    q(:,k+1)=(problem.R+problem.gamma*problem.P*v(:,k+1));
end
q_opt=q(:,N_steps);
v_opt=v(:,N_steps);
end


