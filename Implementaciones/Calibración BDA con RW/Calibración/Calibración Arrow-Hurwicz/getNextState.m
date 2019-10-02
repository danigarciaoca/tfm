function [ nextState, realCurrentAction, reward ] = getNextState( game, currentState, action )
%GETNEXTSTATE Devuelve el siguiente estado y la recompensa instantánea.
%   En base al estado actual (currentState) y a la acción tomada (action),
%   devuelve el siguiente estado y la recompensa asociada a dicha
%   transición, así como la verdadera acción tomada en función de la matriz
%   de transición P

% Obtenemos el siguiente estado tras tomar la acción action desde
% currentState
nextState = discretesample(game.P((currentState-1)*game.N_actions + action,:), 1);

% Obtenemos la reward asociada a la transición del currentState tras tomar la acción action
if action == 1 && nextState==currentState+1 % moverse a la derecha cuando debería a la derecha
    reward = game.R((currentState-1)*game.N_actions + action);
    realCurrentAction = 1;
elseif action == 1 && nextState==currentState-1 % moverse a la izquierda cuando debería a la derecha
    reward = game.R((currentState-1)*game.N_actions + 2);
    realCurrentAction = 2;
elseif action == 2 && nextState==currentState-1 % moverse a la izquierda cuando debería a la izquierda
    reward = game.R((currentState-1)*game.N_actions + action);
    realCurrentAction = 2;
elseif action == 2 && nextState==currentState+1 % moverse a la derecha cuando debería a la izquierda
    reward = game.R((currentState-1)*game.N_actions + 1);
    realCurrentAction = 1;
elseif sum(nextState == game.finalState) == 1
    reward = 0;
end

end