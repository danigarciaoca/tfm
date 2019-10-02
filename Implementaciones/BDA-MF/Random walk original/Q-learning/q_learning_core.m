function [Qsa, Vs, G] = q_learning_core(game, Qsa, epsilon, alpha, numEpisodes, maxNumStepsPerEpisode, init_state)
% Return por episodio
G = zeros(1,numEpisodes);

for i = 1:numEpisodes
    % Inicializamos S
    % currentState = game.centralState;
    currentState = init_state; % empezar en el de la izquierda
    stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpisodes episodes simulated
    
    while true % siempre que el episodio no haya terminado
        % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
        currentAction = e_greedy(Qsa, epsilon, currentState, game.N_actions);
        
        % Tomamos la acción A (currentAction), observamos la recompensa R
        % (reward) y el siguiente estado S' (nextState).
        [nextState, reward] = getNextState(game, currentState, currentAction);
        
        % G(exp,i) = reward+game.gamma*G(exp,i); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
        G(i) = (game.gamma^stepPerEpisode)*reward + G(i);
        
        % Actualizamos el valor de Q(s,a)
        nextAction = greedy(Qsa, nextState);
        Qsa(currentState, currentAction) = Qsa(currentState, currentAction) + alpha*(reward + game.gamma*Qsa(nextState, nextAction) - Qsa(currentState, currentAction));
        
        % Actualizamos los valores
        stepPerEpisode = stepPerEpisode + 1;
        currentState = nextState;
        
        % Evaluación de si el episodio ha terminado o no
        if sum(currentState == game.finalState) == 1 || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
            break; % terminamos el episodio
        end
    end
    % Acumulamos el valor de la función V(s)
    policy = getPolicy(Qsa, game);
    Vs = getValueFunction(Qsa, policy, game);
end
end

