function [ v q] = bellman_linear( problem)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
v=inv(eye(problem.N_states)-problem.gamma*problem.pi_rp*problem.P)*problem.pi_rp*problem.R;
q=inv(eye(problem.N_states*problem.N_actions)-problem.gamma*problem.P*problem.pi_rp)*problem.R;


end

