function [ nextState, reward ] = getNextState( game, currentState, action )
%GETNEXTSTATE Devuelve el siguiente estado y la recompensa instant�nea.
%   En base al estado actual (currentState) y a la acci�n tomada (action),
%   devuelve el siguiente estado y la recompensa asociada a dicha
%   transici�n, as� como la verdadera acci�n tomada en funci�n de la matriz
%   de transici�n P

nextState = discretesample(game.P((currentState-1)*game.N_actions+action,:), 1);
reward = game.Rs(nextState);

% % Obtenemos el siguiente estado tras tomar la acci�n action desde
% % currentState
% nextState = discretesample(game.P((currentState-1)*game.N_actions + action,:), 1);
% 
% % Obtenemos la reward asociada a la transici�n del currentState tras tomar la acci�n action
% if action == 2
%     if nextState==currentState+1 || (nextState==currentState && nextState == game.final_state(2))
%         reward = game.R((currentState-1)*game.N_actions + action); % moverse a la derecha cuando deber�a a la derecha
%     elseif nextState==currentState-1 || (nextState==currentState && nextState == game.final_state(1))
%         reward = game.R((currentState-1)*game.N_actions + 1); % moverse a la izquierda cuando deber�a a la derecha
%     end
% elseif action == 1
%     if nextState==currentState-1 || (nextState==currentState && nextState == game.final_state(1)) % moverse a la izquierda cuando deber�a a la izquierda
%         reward = game.R((currentState-1)*game.N_actions + action);
%     elseif nextState==currentState+1 || (nextState==currentState && nextState == game.final_state(2)) % moverse a la derecha cuando deber�a a la izquierda
%         reward = game.R((currentState-1)*game.N_actions + 2);
%     end
% end

end