function [policy, policy_vector] = GetPolicy(theta, GetStateFeatures, NA, NS)
%GETPOLICY Summary of this function goes here
%   policy: devuelve la política en términos de las acciones (1 y 2) por
%   ejemplo
%   policy vector: devuelve la política en probabilidades (0% o 100%)

q = zeros(1,NA);
policy = zeros(1,NS);
policy_vector = zeros(1, NA*NS);

for s = 1:NS
    phi_s = GetStateFeatures(s);
    for a = 1:NA
        phi_sa = GetStateActionFeatures(phi_s, a, NA);
        q(a) = phi_sa'*theta;
    end
    
    actMax = find(q == max(q));
    action = actMax(randi(length(actMax))); % en caso de empate, seleccionamos una aleatoriamente. Si no hay empate, se elige la única que haya
    
    policy(s) = action;
    policy_vector((s-1)*NA + action) = 1;
end

end