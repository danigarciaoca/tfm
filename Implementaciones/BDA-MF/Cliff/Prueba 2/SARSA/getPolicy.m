function [ Qsa_and_policy_opt, policy_matrix ] = getPolicy( Qsa, game )
%GETPOLICY Función para obtener la policy y la función Q óptimas
%   grid_result -> representación visual del tablero y la acción tomada
%   en cada estado. De nuevo, las celdas con valor NaN representan el
%   acantilado.
%   policy_lineal -> vector columna con la greedy policy de la función Q(s,a) pasada por argumento.
%   Qsa -> función Qsa de la que se quiere conocer la greedy policy.

% Este if sólo entra si la dimensión 1 de Q(s,a) es S*A, lo cual significa
% que está recibiendo la función Q(s,a) en forma lineal para hacer el
% promedio a lo largo de todos los experimentos.
if size(Qsa,1) == game.numActions*game.numStates
    % Calculamos la media del último valor de Q(s,a) que se obtuvo en cada experimento
    Qsa = mean(Qsa(:,end,:),3);
    Qsa = reshape(Qsa, [game.numActions game.numStates])';
end
[Qsa_optima, policy_lineal] = max(Qsa,[],2);

policy_matrix = zeros(game.numStates, game.numActions*game.numStates);
for st = 1:game.numStates
    act_max = find(Qsa_optima(st) == Qsa(st,:)==1);

    if size(act_max,2) > 1 % Si hay más de una acción que maximiza, escoger una aleatoriamente
        %act_max = randi(act_max);
        act_max = randsample(act_max,1);
    end
    policy_matrix(st, (st-1)*game.numActions + act_max) = 1;
end

Qsa_and_policy_opt.Qsa = Qsa;
Qsa_and_policy_opt.Qsa_optima = Qsa_optima;
Qsa_and_policy_opt.pi_lineal = policy_lineal;
end

