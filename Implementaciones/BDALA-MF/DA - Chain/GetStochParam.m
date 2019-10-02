function [ theta_opt ] = GetStochParam( reward_t, phi_t, phi_t1, gamma)
%GETEXACTPARAMOPT This function returns the exact and optimum calculation
%   of vector of state parameters theta, given the reward vector R, transition 
%   probability matrix P and features/basis functions matrix phi

theta_opt = inv(phi_t'*phi_t - gamma*phi_t'*phi_t1) * phi_t'*reward_t;

end

