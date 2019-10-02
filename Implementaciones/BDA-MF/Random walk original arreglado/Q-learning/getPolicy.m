function policy_matrix = getPolicy( Qsa, game )
%GETPOLICY Función para obtener la policy óptima a partir de la función Q óptima
%   policy -> greedy policy de la función Q(s,a) pasada por argumento.
%   Qsa -> función Qsa de la que se quiere conocer la greedy policy.

if numel(size(Qsa)) == 3 || size(Qsa,2) > 2
    % Calculamos la media del último valor de Q(s,a) que se obtuvo en cada experimento
    Qsa = mean(Qsa(:,end,:),3);
    Qsa = reshape(Qsa, [game.numActions game.numStates])';
end
Qsa_optima = max(Qsa,[],2);

policy_matrix = zeros(game.numStates, game.numActions*game.numStates);
for st = 1:game.numStates
    act_max = find(Qsa_optima(st) == Qsa(st,:)==1);
    if size(act_max,2) >2
        disp('pausa')
    end
    if size(act_max,2) > 1 % Si hay más de una acción que maximiza, escoger una aleatoriamente
        act_max = randi(act_max);
    end
    policy_matrix(st, (st-1)*game.numActions + act_max) = 1;
end
end