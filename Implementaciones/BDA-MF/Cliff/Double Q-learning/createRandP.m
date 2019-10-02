function [R, transitions] = createRandP(S,A, env)
rewards = zeros(S, A);
transitions = zeros(S*A,S);
for currentState = 1:S
    isCliffCurrent = length(find(currentState == env.cliff)) == 1;
    isTerminalCurrent = (currentState == env.finalState);

    % Si el estado actual no es cliff ni terminal, proceso normal
    % Si el estado actual es cliff o terminal, no entramos al bucle
    if ~(isCliffCurrent || isTerminalCurrent)
        for action = 1:A
            nextState = currentState;
            wall = sum((currentState == env.walls(end,:)) & env.walls(action, :));
            if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                [rPos, cPos] = find(env.board==currentState); % Determinamos nuestra posición relativa a las coordenadas del tablero
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
                nextState = env.board(rPos, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
            end
            isCliffNext = length(find(nextState == env.cliff)) == 1;
            % Una vez conocemos el nuevo estado, devolvemos la recompensa pertinente
            if isCliffNext % Si el estado al que nos hemos movido es cliff
                reward = env.rewardOnCliff; % Recompensa de haber caído en cliff
            elseif nextState == env.finalState % Si el estado al que nos hemos movido es el terminal
                reward = env.rewardOnTransition; % Recompensa de una transición cualquiera
            else % Si el estado al que nos hemos movido es cualquiera distinto del cliff y no estabamos en el estado final
                reward = env.rewardOnTransition; % Recompensa de una transición cualquiera
            end
            
            rewards(currentState, action) = reward;
            transitions((currentState-1)*A + action, nextState) = 1;
        end
    end
end
R = reshape(rewards', [S*A 1]);
end