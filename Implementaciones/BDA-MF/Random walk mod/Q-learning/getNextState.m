function [ nextState, reward ] = getNextState( game, currentState, action )
%GETNEXTSTATE Devuelve el siguiente estado y la recompensa instant�nea.
%   En base al estado actual (currentState) y a la acci�n tomada (action),
%   devuelve el siguiente estado y la recompensa asociada a dicha
%   transici�n.

% Obtenemos el siguiente estado tras tomar la acci�n action desde
% currentState
nextState = discretesample(game.P((currentState-1)*game.N_actions + action,:), 1);

% Obtenemos la reward asociada a la transici�n del currentState tras tomar la acci�n action
if action == 1 && nextState==currentState+1 % moverse a la derecha cuando deber�a a la derecha
    reward = game.R((currentState-1)*game.N_actions + action);
elseif action == 1 && nextState==currentState-1 % moverse a la izquierda cuando deber�a a la derecha
    reward = game.R((currentState-1)*game.N_actions + 2);
elseif action == 2 && nextState==currentState-1 % moverse a la izquierda cuando deber�a a la izquierda
    reward = game.R((currentState-1)*game.N_actions + action);
elseif action == 2 && nextState==currentState+1 % moverse a la derecha cuando deber�a a la izquierda
    reward = game.R((currentState-1)*game.N_actions + 1);
end

end