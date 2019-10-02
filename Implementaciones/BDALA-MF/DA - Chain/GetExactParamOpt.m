function [ theta_opt ] = GetExactParamOpt( R, P, phi, gamma)
%GETEXACTPARAMOPT This function returns the exact and optimum calculation
%   of vector of state parameters theta, given the reward vector R, transition 
%   probability matrix P and features/basis functions matrix phi

D = GetLimStationStateVisitProb(P);
theta_opt = inv(phi'*D*phi - gamma*phi'*D*P*phi) * phi'*D*R;

end

