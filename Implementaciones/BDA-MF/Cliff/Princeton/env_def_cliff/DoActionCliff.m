function [ snew, reward, terminal ] = DoActionCliff( action, currentState, env )
%GETNEXTSTATE Devuelve el siguiente estado y la recompensa instant�nea.
%   En base al estado actual (currentState) y a la acci�n tomada (action),
%   devuelve el siguiente estado y la recompensa asociada a dicha
%   transici�n.

terminal = false; % flag que indica si el episodio ha terminado
Pssa = env.P;
R = env.R;

snew = discretesample(Pssa((currentState-1)*env.numActions + action,:), 1); % Observe new state

reward = R(snew); % Observe reward for the new state

isCliffNext = (length(find(snew == env.cliff)) == 1);
if isCliffNext || (snew == env.finalState)
    terminal = true;
end

end

% VERSI�N VIEJA
% % Determinamos si, tomando la acci�n pedida en el estado actual, vamos a
% % chocarnos contra una pared (wall = 1) o no (wall = 0)
% wall = sum((currentState == env.walls(end,:)) & env.walls(action, :));
%
% % Si S� nos vamos a chocar, no actualizamos la posici�n y mantenemos la que ten�amos
%
% if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
%     [rPos, cPos] = find(env.board==currentState); % Determinamos nuestra posici�n relativa a las coordenadas del tablero
%     switch action
%         case 1 % izquierda
%             cPos = cPos-1;
%         case 2 % arriba
%             rPos = rPos-1;
%         case 3 % derecha
%             cPos = cPos+1;
%         case 4 % abajo
%             rPos = rPos+1;
%     end
%     currentState = env.board(rPos, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
% end
% % Comprobamos si el siguiente estado es cliff
% isCliffNext = length(find(currentState == env.cliff)) == 1;
%
% % Una vez conocemos el nuevo estado, devolvemos la recompensa pertinente
% if isCliffNext % Si el estado al que nos hemos movido es cliff
%     reward = env.rewardOnCliff; % Recompensa de haber ca�do en cliff
%     terminal = true; % terminamos el episodio
% elseif currentState == env.finalState % Si el estado al que nos hemos movido es el terminal
%     % reward = 0; % Recompensa cero (fin del juego)
%     reward = env.rewardOnTransition; % Recompensa de una transici�n cualquiera
%     terminal = true; % terminamos el episodio
% else % Si el estado al que nos hemos movido es cualquiera distinto del cliff y no estabamos en el estado final
%     reward = env.rewardOnTransition; % Recompensa de una transici�n cualquiera
% end