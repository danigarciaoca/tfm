function [ nextState, reward ] = getNextState( game, currentState, action )
%GETNEXTSTATE Devuelve el siguiente estado y la recompensa instantánea.
%   En base al estado actual (currentState) y a la acción tomada (action),
%   devuelve el siguiente estado y la recompensa asociada a dicha
%   transición.

% Obtenemos el siguiente estado tras tomar la acción action desde
% currentState
nextState = discretesample(game.P((currentState-1)*game.N_actions + action,:), 1);

% Obtenemos la reward asociada a la transición del currentState tras tomar la acción action
if action == 1 && nextState==currentState+1 % moverse a la derecha cuando debería a la derecha
    reward = game.R((currentState-1)*game.N_actions + action);
elseif action == 1 && nextState==currentState-1 % moverse a la izquierda cuando debería a la derecha
    reward = game.R((currentState-1)*game.N_actions + 2);
elseif action == 2 && nextState==currentState-1 % moverse a la izquierda cuando debería a la izquierda
    reward = game.R((currentState-1)*game.N_actions + action);
elseif action == 2 && nextState==currentState+1 % moverse a la derecha cuando debería a la izquierda
    reward = game.R((currentState-1)*game.N_actions + 1);
end

end