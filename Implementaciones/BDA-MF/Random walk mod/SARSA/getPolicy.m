function policy_matrix = getPolicy( Qsa, game )
%GETPOLICY Funci�n para obtener la policy �ptima a partir de la funci�n Q �ptima
%   policy -> greedy policy de la funci�n Q(s,a) pasada por argumento.
%   Qsa -> funci�n Qsa de la que se quiere conocer la greedy policy.

if numel(size(Qsa)) == 3 || size(Qsa,2) > 2
    % Calculamos la media del �ltimo valor de Q(s,a) que se obtuvo en cada experimento
    Qsa = mean(Qsa(:,end,:),3);
    Qsa = reshape(Qsa, [game.N_actions game.N_states])';
end
Qsa_optima = max(Qsa,[],2);

policy_matrix = zeros(game.N_states, game.N_actions*game.N_states);
for st = 1:game.N_states
    act_max = find(Qsa_optima(st) == Qsa(st,:)==1);
    if size(act_max,2) >2
        disp('pausa')
    end
    if size(act_max,2) > 1 % Si hay m�s de una acci�n que maximiza, escoger una aleatoriamente
        act_max = randi(act_max);
    end
    policy_matrix(st, (st-1)*game.N_actions + act_max) = 1;
end
end