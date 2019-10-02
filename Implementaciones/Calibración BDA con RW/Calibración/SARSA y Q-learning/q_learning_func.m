function q_learning_func( numExperiments, numEpisodes, epsilon, alpha)
%Q_LEARNING_FUNC Summary of this function goes here
%   Detailed explanation goes here

load juego_test.mat

% Optimum value
v_opt=inv(eye(game.N_states)-game.gamma*game.pi_opt*game.P)*game.pi_opt*game.R;
q_opt=inv(eye(game.N_states*game.N_actions)-game.gamma*game.P*game.pi_opt)*game.R;

% % % % AGENTE:
% numExperiments = 50; % Número de experimentos sobre el que se promedia
% numEpisodes = 5000; % Número de episodios de cada experimento
maxNumStepsPerEpisode = 500; % Número máximo de pasos en cada episodio
G = zeros(numExperiments,numEpisodes); % Reward por episodio
% epsilon = 0.1; % e-greedy value 
% alpha = 0.3; %SUBIR (HACER PRUEBA CON 0.3)

% Variable que acumulará las funciones Q y V al final de cada episodio
Qsa_lineal_acumulada = zeros(game.N_states*game.N_actions, numEpisodes, numExperiments);
Vs_acumulada = zeros(game.N_states, numEpisodes, numExperiments);

for exp = 1:numExperiments
    % Inicializamos Q para cada experimento
    Qsa = rand(game.N_states, game.N_actions);
    Qsa(1,:) = zeros(size(Qsa(1,:)));
    Qsa(end,:) = zeros(size(Qsa(1,:)));
    
    for i = 1:numEpisodes
        % Inicializamos S
        % currentState = game.centralState;
        currentState = 2; % empezar en el de la izquierda
        stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpisodes episodes simulated
        
        while true % siempre que el episodio no haya terminado
            % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
            currentAction = e_greedy(Qsa, epsilon, currentState, game.N_actions);
            
            % Tomamos la acción A (currentAction), observamos la recompensa R
            % (reward) y el siguiente estado S' (nextState).
            [nextState, reward] = getNextState(game, currentState, currentAction);
            % G(exp,i) = reward+game.gamma*G(exp,i); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
            G(exp,i) = (game.gamma^stepPerEpisode)*reward+G(exp,i);
            
            % Actualizamos el valor de Q(s,a)
            nextAction = greedy(Qsa, currentState);
            Qsa(currentState, currentAction) = Qsa(currentState, currentAction) + alpha*(reward + game.gamma*Qsa(nextState, nextAction) - Qsa(currentState, currentAction));
            
            % Actualizamos los valores
            stepPerEpisode = stepPerEpisode + 1;
            currentState = nextState;
            
            % Evaluación de si el episodio ha terminado o no
            if sum(currentState == game.finalState) == 1 || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
                break; % terminamos el episodio
            end
%             [Qsa reshape(q_opt,[2,7])']
        end
        % Acumulamos el valor de la función Q(s,a)
        Qsa_lineal_acumulada(:, i, exp) = reshape(Qsa', [game.N_states*game.N_actions 1]);
        
        % Acumulamos el valor de la función V(s)
        policy = getPolicy(Qsa, game);
        Vs_acumulada(:, i, exp) = getValueFunction(Qsa, policy, game);
    end
end

% getPolicy obtiene la policy del promedio de las Qsa de todos los
% experimentos realizados
policy = getPolicy(Qsa_lineal_acumulada, game);
Qsa;
Vs = getValueFunction(Qsa, policy, game);

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
% figure, plot((1:numEpisodes), Gmean, 'LineWidth', 2)
% title(['Reward per episode (always starting at s=' ,num2str(game.centralState), ')'])
% xlabel('Episode'), ylabel('G')

aux1 = sum(Qsa_lineal_acumulada);
aux2 = squeeze(aux1)';
sumQ_mean = mean(aux2);
% figure, plot((1:numEpisodes), sumQ_mean, 'LineWidth', 2)
% title('\Sigmaq(s,a) for each episode')
% xlabel('Episode'), ylabel('\Sigmaq(s,a)')

aux1 = sum(Vs_acumulada);
aux2 = squeeze(aux1)';
sumV_mean = mean(aux2);
% figure, plot((1:numEpisodes), sumV_mean, 'LineWidth', 2)
% title('\Sigmav(s) for each episode')
% xlabel('Episode'), ylabel('\Sigmav(s)')

aux1 = sum((Qsa_lineal_acumulada(game.N_actions+1:end-game.N_actions,:,:)-q_opt(game.N_actions+1:end-game.N_actions,:)).^2 , 1);
aux2 = squeeze(aux1)';
mse_q = sqrt(mean(aux2));
% figure, plot((1:numEpisodes), mse_q, 'LineWidth', 2)
% title('Mean-squared error of q(s,a) for each episode')
% xlabel('Episode'), ylabel('MSE q(s,a)')

aux1 = sum((Vs_acumulada(2:end-1,:,:)-v_opt(2:end-1,:)).^2 , 1);
aux2 = squeeze(aux1)';
mse_v = sqrt(mean(aux2));
% figure, plot((1:numEpisodes), mse_v, 'LineWidth', 2)
% title('Mean-squared error of v(s) for each episode')
% xlabel('Episode'), ylabel('MSE v(s)')

% Calcula el error en la policy en cada episodio, promediado entre todos
% los experimentos
policy_vector = getPolicyVector(Qsa_lineal_acumulada, game); % policy de cada experimento y para cada iteración
optimal_policy_vector = getPolicyVector(q_opt, game); % policy óptima
errorPolicy = policy_vector - optimal_policy_vector; % error de policies
dif = policy_vector(game.N_actions+1:end-game.N_actions,:,:) - optimal_policy_vector(game.N_actions+1:end-game.N_actions);
policy_error_mean = mean(sqrt(sum(dif.^2)),3); % ecm y promedio de entre todos los experimentos

% figure, plot((1:numEpisodes), policy_error_mean, 'LineWidth', 2)
% title('Policy error')
% xlabel('Episode'), ylabel('d-d_{opt}')

% [Vs v_opt]
% [Qsa_lineal_acumulada(:,end) q_opt]
% sum(policy(:,3:end-2)) % comprobación de la policy; deben estar los unos alternos
save(['Resultados\numExperiments=' num2str(numExperiments) ',numEpisodes=' num2str(numEpisodes) ',alpha=' num2str(alpha) ',epsilon=' num2str(epsilon) '.mat'])
end

