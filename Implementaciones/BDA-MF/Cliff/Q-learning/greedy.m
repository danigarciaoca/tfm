function action = greedy( Qsa, currentState )
%GREEDY Implementación de la greedy policy
%   Tomamos siempre la acción que maximiza Q(s,a)

% Para el estado actual, buscamos las acciones que maximizan Q(s,a)
actMax = find(Qsa(currentState, :) == max(Qsa(currentState, :)));
action = actMax(randi(length(actMax))); % en caso de empate, seleccionamos una aleatoriamente. Si no hay empate, se elige la única que haya

end

