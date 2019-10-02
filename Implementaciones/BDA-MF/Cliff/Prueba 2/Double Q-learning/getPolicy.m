function [ Qsa_and_policy_opt, policy_matrix ] = getPolicy( Qsa, game )
%GETPOLICY Funci�n para obtener la policy y la funci�n Q �ptimas
%   grid_result -> representaci�n visual del tablero y la acci�n tomada
%   en cada estado. De nuevo, las celdas con valor NaN representan el
%   acantilado.
%   policy_lineal -> vector columna con la greedy policy de la funci�n Q(s,a) pasada por argumento.
%   Qsa -> funci�n Qsa de la que se quiere conocer la greedy policy.

% Este if s�lo entra si la dimensi�n 1 de Q(s,a) es S*A, lo cual significa
% que est� recibiendo la funci�n Q(s,a) en forma lineal para hacer el
% promedio a lo largo de todos los experimentos.
if size(Qsa,1) == game.numActions*game.numStates
    % Calculamos la media del �ltimo valor de Q(s,a) que se obtuvo en cada experimento
    Qsa = mean(Qsa(:,end,:),3);
    Qsa = reshape(Qsa, [game.numActions game.numStates])';
end
[Qsa_optima, policy_lineal] = max(Qsa,[],2);

policy_matrix = zeros(game.numStates, game.numActions*game.numStates);
for st = 1:game.numStates
    act_max = find(Qsa_optima(st) == Qsa(st,:)==1);

    if size(act_max,2) > 1 % Si hay m�s de una acci�n que maximiza, escoger una aleatoriamente
        %act_max = randi(act_max);
        act_max = randsample(act_max,1);
    end
    policy_matrix(st, (st-1)*game.numActions + act_max) = 1;
end

Qsa_and_policy_opt.Qsa = Qsa;
Qsa_and_policy_opt.Qsa_optima = Qsa_optima;
Qsa_and_policy_opt.pi_lineal = policy_lineal;
end

