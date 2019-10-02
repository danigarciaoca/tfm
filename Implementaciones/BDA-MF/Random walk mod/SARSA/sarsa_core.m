function [Qsa, Vs, G] = sarsa_core(game, Qsa, epsilon, alpha, numEpisodes, maxNumStepsPerEpisode, init_state)
% Return por episodio
G = zeros(1,numEpisodes);

for i = 1:numEpisodes
    % Inicializamos S
    % currentState = game.centralState;
    currentState = init_state; % empezar en el de la izquierda
    stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpisodes episodes simulated
    
    % Escogemos A (currentAction) de S (currentState) seg�n la e-greedy policy.
    currentAction = e_greedy(Qsa, epsilon, currentState, game.N_actions);
    
    while true % siempre que el episodio no haya terminado
        
        % Tomamos la acci�n A (currentAction), observamos la recompensa R
        % (reward) y el siguiente estado S' (nextState).
        [nextState, reward] = getNextState(game, currentState, currentAction);
        
        % G(exp,i) = reward+game.gamma*G(exp,i); % return following the initial state (si el estado inicial fuese aleatorio, habr�a que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
        G(i) = (game.gamma^stepPerEpisode)*reward + G(i);
        
        % Escogemos A' (nextAction) de S' (nextState) seg�n la e-greedy policy.
        nextAction = e_greedy(Qsa, epsilon, nextState, game.N_actions);
        
        % Actualizamos el valor de Q(s,a)
        %Qsa(currentState, currentAction) = Qsa(currentState, currentAction) + alpha*(reward + game.gamma*Qsa(nextState, nextAction) - Qsa(currentState, currentAction));
        
        % Actualizamos los valores
        currentState = nextState;
        currentAction = nextAction;
        stepPerEpisode = stepPerEpisode + 1;
        
        % Evaluaci�n de si el episodio ha terminado o no
        if sum(currentState == game.finalState) == 1 || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
            break; % terminamos el episodio
        end
    end
    % Acumulamos el valor de la funci�n V(s)
    policy = getPolicy(Qsa, game);
    Vs = getValueFunction(Qsa, policy, game);
end
end

