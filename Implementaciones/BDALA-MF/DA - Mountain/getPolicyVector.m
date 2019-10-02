function policy_vector = getPolicyVector(Qsa_lineal_acumulada, game)
%GETPOLICYVECTOR Summary of this function goes here
%   Detailed explanation goes here

policy_vector = zeros(game.num_actions*game.S, size(Qsa_lineal_acumulada,2), size(Qsa_lineal_acumulada,3));
for i = 1:size(Qsa_lineal_acumulada,3) % choose experiment
    for j = 1:size(Qsa_lineal_acumulada,2) % choose episode
        Qsa = reshape(Qsa_lineal_acumulada(:,j,i), [game.num_actions game.S])';
        Qsa_optima = max(Qsa,[],2);
        
        for st = 1:game.S
            act_max = find(Qsa_optima(st) == Qsa(st,:)==1);
            if size(act_max,2) > 1 % Si hay más de una acción que maximiza, escoger una aleatoriamente
                act_max = randi(act_max);
            end
            policy_vector(((st-1)*game.num_actions)+act_max, j,i) = 1;
        end
    end
end
end

