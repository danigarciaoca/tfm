function [ currentState, reward ] = getNextState( game, currentState, action )
%GETNEXTSTATE Devuelve el siguiente estado y la recompensa instant�nea.
%   En base al estado actual (currentState) y a la acci�n tomada (action),
%   devuelve el siguiente estado y la recompensa asociada a dicha
%   transici�n.

% Determinamos si, tomando la acci�n pedida en el estado actual, vamos a
% chocarnos contra una pared (wall = 1) o no (wall = 0)
wall = sum((currentState == game.walls(end,:)) & game.walls(action, :));

% Determinamos si estamos en el estado final (1) o no (0)
final = currentState==game.finalState;

% Si S� nos vamos a chocar o S� estamos en el estado final, no actualizamos la
% posici�n y mantenemos la que ten�amos
wallOrFinal = wall | final;

if wallOrFinal == 0 % Si NO nos vamos a chocar y NO estamos en el estado final, nos movemos al siguiete estado
    [rPos, cPos] = find(game.board==currentState); % Determinamos nuestra posici�n relativa a las coordenadas del tablero
    switch action
        case 1 % izquierda
            cPos = cPos-1;
        case 2 % arriba
            rPos = rPos-1;
        case 3 % derecha
            cPos = cPos+1;
        case 4 % abajo
            rPos = rPos+1;
    end
    currentState = game.board(rPos, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
end

% Una vez conocemos el nuevo estado, devolvemos la recompensa pertinente
if isnan(currentState) % Si el estado al que nos hemos movido vale NaN, es que hemos ca�do en el cliff
    currentState = game.initState; % Volvemos al estado inicial
    reward = game.rewardOnCliff; % Recompensa de haber ca�do en cliff
elseif final == 1 % Si estabamos en el estado final
    reward = 0; % Recompensa cero (fin del juego)
else % Si el estado al que nos hemos movido es cualquiera distinto del cliff y no estabamos en el estado final
    reward = game.rewardOnTransition; % Recompensa de una transici�n cualquiera
end

end