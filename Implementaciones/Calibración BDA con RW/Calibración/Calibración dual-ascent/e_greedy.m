function action = e_greedy( d_norm, epsilon, currentState, numActions )
%E_GREEDY Implementación de la policy e-greedy
%   A través de una v.a de Bernoulli, conseguimos que:
%   Con probabilidad 1-epsilon exploTemos el espacio de estados
%   Con probabilidad epsilon exploRemos el espacio de estados

% Con probabilidad epsilon hay exito (1)
% Con probabilidad 1-epsilon hay fracaso (0)
R = binornd(1,epsilon);

if R == 0 % Con probabilidad 1-epsilon exploTamos
      action = discretesample(d_norm((((currentState-1)*numActions)+1):currentState*numActions), 1);
%     % Para el estado actual, buscamos las acciones que maximizan Q(s,a)
%     actMax = find(Qsa(currentState, :) == max(Qsa(currentState, :)));
%     action = actMax(randi(length(actMax))); % en caso de empate, seleccionamos una aleatoriamente. Si no hay empate, se elige la única que haya
elseif R == 1 % Con probabilidad epsilon exploRamos
    action = randi(numActions); % escogemos una acción de entre las posibles de manera aleatoria
end

end

