function policy = GetPolicy(theta, GetStateFeatures, NA, NS)
%GETPOLICY Summary of this function goes here
%   Detailed explanation goes here

q = zeros(1,NA);
policy = zeros(1,NS);

for s = 1:NS
    phi_s = GetStateFeatures(s);
    for a = 1:NA
        phi_sa = GetStateActionFeatures(phi_s, a, NA);
        q(a) = phi_sa'*theta;
    end
    
    actMax = find(q == max(q));
    action = actMax(randi(length(actMax))); % en caso de empate, seleccionamos una aleatoriamente. Si no hay empate, se elige la única que haya
    
    policy(s) = action;
end

end