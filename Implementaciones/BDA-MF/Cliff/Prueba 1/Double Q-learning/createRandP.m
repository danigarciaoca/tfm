function [R, transitions] = createRandP(S,A, env, transitionType)
rewards = zeros(S, A);
transitions = zeros(S*A,S);
for currentState = 1:S
    isCliffCurrent = length(find(currentState == env.cliff)) == 1;
    isTerminalCurrent = (currentState == env.finalState);
    
    % Si el estado actual no es cliff ni terminal, proceso normal
    % Si el estado actual es cliff o terminal, no entramos al bucle
    if ~(isCliffCurrent || isTerminalCurrent)
        for action = 1:A
            if strcmp(transitionType, 'rand')
                min_p_good = 0.6;
                p_good = min_p_good+(1-min_p_good)*rand;
                p_bad = (1-p_good)/3;
            elseif strcmp(transitionType, 'det')
                p_good = 1;
                p_bad = (1-p_good)/3;
            end
            
            [rPos, cPos] = find(env.board==currentState); % Determinamos nuestra posición relativa a las coordenadas del tablero
            switch action
                case 1 % izquierda
                    % izquierda (movimiento esperado)
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(1, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos, cPos-1); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_good + transitions((currentState-1)*A + action, nextState);
                    
                    % arriba
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(2, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos-1, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % derecha
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(3, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos, cPos+1); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % abajo
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(4, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos+1, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % Dejamos constancia de la acción verdadera tomada (izquierda)
                    wall = sum((currentState == env.walls(end,:)) & env.walls(action, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        cPos = cPos-1;
                    end
                    % Si nos ibamos a chocar, no actualizamos la posición
                    
                case 2 % arriba
                    % izquierda
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(1, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos, cPos-1); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % arriba (movimiento esperado)
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(2, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos-1, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_good + transitions((currentState-1)*A + action, nextState);
                    
                    % derecha
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(3, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos, cPos+1); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % abajo
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(4, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos+1, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % Dejamos constancia de la acción verdadera tomada (arriba)
                    wall = sum((currentState == env.walls(end,:)) & env.walls(action, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        rPos = rPos-1;
                    end
                    % Si nos ibamos a chocar, no actualizamos la posición
                    
                case 3 % derecha
                    % izquierda
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(1, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos, cPos-1); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % arriba
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(2, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos-1, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % derecha (movimiento esperado)
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(3, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos, cPos+1); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_good + transitions((currentState-1)*A + action, nextState);
                    
                    % abajo
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(4, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos+1, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % Dejamos constancia de la acción verdadera tomada (derecha)
                    wall = sum((currentState == env.walls(end,:)) & env.walls(action, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        cPos = cPos+1;
                    end
                    % Si nos ibamos a chocar, no actualizamos la posición
                    
                case 4 % abajo
                    % izquierda
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(1, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos, cPos-1); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % arriba
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(2, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos-1, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % derecha
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(3, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos, cPos+1); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_bad + transitions((currentState-1)*A + action, nextState);
                    
                    % abajo (movimiento esperado)
                    nextState = currentState;
                    wall = sum((currentState == env.walls(end,:)) & env.walls(4, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        nextState = env.board(rPos+1, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
                    end
                    transitions((currentState-1)*A + action, nextState) = p_good + transitions((currentState-1)*A + action, nextState);
                    
                    % Dejamos constancia de la acción verdadera tomada (abajo)
                    wall = sum((currentState == env.walls(end,:)) & env.walls(action, :));
                    if wall == 0 % Si NO nos vamos a chocar, nos movemos al siguiete estado
                        rPos = rPos+1;
                    end
                    % Si nos ibamos a chocar, no actualizamos la posición
            end
            nextState = env.board(rPos, cPos); % Obtenemos el nuevo estado actual (al que nos hemos movido)
            
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
        end
    end
end
R = reshape(rewards', [S*A 1]);
end