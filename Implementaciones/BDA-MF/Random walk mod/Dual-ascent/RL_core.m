function [d, Vs, G] = RL_core(game, d, v, epsilon, alphaTD, alphaD, numRep, numEpisodes, maxNumStepsPerEpisode, mult1, mult2, init_state)
% Asignaciones por comodidad
S = game.N_states;
A = game.N_actions;
% Return por episodio
G = zeros(1,numRep*numEpisodes);
Vs = zeros(S,numRep*numEpisodes);

episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
for k = 1:numRep
    s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
    totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
    
    for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
        % currentState = game.centralState; % estado inicial el central
        % currentState = randi([2 game.centralState]);
        currentState = init_state; % empezar en el de la izquierda
        terminal = false; % true when episode finish, false otherwise
        stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
        
        % Normalize d
        d_norm = getPolicyVectorFromD( d, game );
        % Get policy matrix
        policy_by_action = reshape(d_norm', [A,S])';
        policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';
        while ~terminal
            % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
            currentAction = e_greedy(d_norm, epsilon, currentState, game.N_actions);
            
            % Tomamos la acción a (currentAction), observamos la recompensa
            % r (reward) y el siguiente estado s' (nextState).
            [nextState, ~, reward] = getNextState(game, currentState, currentAction);
            % G(exp, (k-1)*numEpisodes+n) = reward+game.gamma*G(exp,(k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
            G((k-1)*numEpisodes+n) = (game.gamma^stepPerEpisode)*reward+G((k-1)*numEpisodes+n);
            
            % Update de v(s)
            % TEMPORAL DIFFERENCE
            v(currentState) = v(currentState) + alphaTD*(reward + game.gamma*v(nextState) - v(currentState)); % policy evaluation
            % EXACTA (BELLMAN)
            % v = (inv(eye(S)-game.gamma*policy_matrix*P))*policy_matrix*R;

            % Actualizamos valores
            stepPerEpisode = stepPerEpisode + 1;
            totalStepsPerRep = totalStepsPerRep + 1;
            currentState = nextState;
            
            % Evaluación de si el episodio ha terminado o no
            if sum(currentState == game.finalState) == 1 || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
                terminal = true;
                episodeCountV = episodeCountV + 1;
                Vs(:, episodeCountV) = v;
                % disp(['Fin' num2str(n) ' y ' num2str(stepPerEpisode)])
            end
        end
    end 
end
end

