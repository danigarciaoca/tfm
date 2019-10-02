clear all, clc

% % % % ENTORNO:
game = ChainWalkSetUp();

A = eye(game.N_states);
B = [1; 0];
mult1 = kron(A,B);
B = [0; 1];
mult2 = kron(A,B);

% % % % AGENTE:
numExperiments = 1; % Número de experimentos de numRep*numEpisodes episodios
numRep = 1000; % Número de repeticiones de cada set de episodios
numEpisodes = 1000; % Número de episodios de cada repeticion
maxNumStepsPerEpisode = 100; % Número máximo de pasos en cada episodio
G = zeros(numExperiments, numRep*numEpisodes); % Reward por episodio
epsilon = 0; % e-greedy value (entre 0.05 y 0.2)
alphaD = 0.1; % Stepsize para la iteración de la variable dual d
alphaTD = 0.4; % Stepsize para la iteración de la variable primal v
% con alphaTD = 0.5 converge más rápido la return

S = game.N_states; % Número de estados
A = game.N_actions; % Número de acciones
N = game.N_features; % states feature vector length
mu = game.mu; % Distribución inicial de probabilida de los estados
P = game.P; % Matriz de transiciones
R = game.R; % Vector de rewards
gamma = game.gamma; % Discount rate/factor
terminal = false; % flag used when terminal state reached

% Optimum values
v_opt = inv(eye(S)-game.gamma*game.pi_opt*P)*game.pi_opt*R;
q_opt = inv(eye(S*A)-game.gamma*P*game.pi_opt)*R;
% Valor óptimo de d (política en forma vector)
d_opt_norm = getPolicyVector(q_opt, game); % policy óptima

% Variable que acumulará la funcion V y el error en la política al final de cada episodio
Vs_acumulada = nan(game.N_states, numRep*numEpisodes, numExperiments);
errorD = nan(numExperiments, numRep*numEpisodes);
d_norm_acumulada = nan(game.N_states*game.N_actions, numRep*numEpisodes, numExperiments);

phi = GetFeatureMatrix(S, N);

for exp = 1:numExperiments
    not_error = true;
    % Inicializamos D y V
    d = rand(S*A,1);
    counter = zeros(S,1);
    episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
  
    for k = 1:numRep
        s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
        totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
        phi_t = nan(numEpisodes*maxNumStepsPerEpisode, N);
        phi_t1 = nan(numEpisodes*maxNumStepsPerEpisode, N);
        reward_t = nan(numEpisodes*maxNumStepsPerEpisode, 1);
        
        for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
            % currentState = game.centralState; % estado inicial el central
            % currentState = randi([2 game.centralState]);
            currentState = game.initial_state; % empezar en el de la izquierda
            terminal = false; % true when episode finish, false otherwise
            stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
            
            % Normalize d
            d_norm = getPolicyVectorFromD( d, game );
            % Get policy matrix
            policy_by_action = reshape(d_norm', [A,S])';
            policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';
            while ~terminal
                counter(currentState) = counter(currentState) + 1;
                % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
                currentAction = e_greedy(d_norm, epsilon, currentState, A);
                
                % Tomamos la acción a (currentAction), observamos la recompensa
                % r (reward) y el siguiente estado s' (nextState).
                [nextState, reward] = getNextState(game, currentState, currentAction);
                % G(exp, (k-1)*numEpisodes+n) = reward+game.gamma*G(exp,(k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
                G(exp, (k-1)*numEpisodes+n) = (game.gamma^stepPerEpisode)*reward+G(exp,(k-1)*numEpisodes+n);
                
                % LEAST-SQUARES TEMPORAL DIFFERENCE (I)
                phi_t(totalStepsPerRep,:) = GetPolyFeatures(currentState);
                phi_t1(totalStepsPerRep,:) = GetPolyFeatures(nextState);
                reward_t(totalStepsPerRep,:) = reward; 

%                 % EXACTA (APROXIMACIÓN DE FUNCIONES)
%                 theta = GetExactParamOpt( policy_matrix*R, policy_matrix*P, phi, gamma);
%                 v = phi*theta;
                
                % Actualizamos valores
                stepPerEpisode = stepPerEpisode + 1;
                totalStepsPerRep = totalStepsPerRep + 1;
                currentState = nextState;
                
                % Evaluación de si el episodio ha terminado o no
                if stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
                    terminal = true;
                    episodeCountV = episodeCountV + 1;
                    % Vs_acumulada(:, episodeCountV, exp) = v;
                    % disp(['Fin' num2str(n) ' y ' num2str(stepPerEpisode)])
                end
            end
        end
        totalStepsPerRep = totalStepsPerRep-1; % Compensamos el que se incrementó de más
        % LEAST-SQUARES TEMPORAL DIFFERENCE (II)
        theta = GetStochParam( reward_t, phi_t, phi_t1, gamma);
        v = phi*theta;
    end
    D = GetLimStationStateVisitProb(policy_matrix*P);
end