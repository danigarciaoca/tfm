function phi_sa = GetStateActionFeatures(phi_s, a, num_actions)

N = length(phi_s);
M = N*num_actions;

phi_sa = zeros(M, 1);
phi_sa((a-1)*N+1:a*N) = phi_s;


end % function getStateActionFeatures