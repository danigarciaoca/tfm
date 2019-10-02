function RL_function(numExperiments, numRep, numEpisodes, alphaD, alphaTD, epsilon)

% % % % ENTORNO:
% save('juego_test.mat','game')
load juego_test.mat
% Policy óptima
[v_opt, q_opt, ~] = getOptimalPolicy(game);

% Policy random
% [ actions, d_rand_norm ] = generateRandomPolicy(game); % actions, numel(find(actions(2:end-1)==1))
% save('random_policy.mat','d_rand_norm')
% [v_opt, q_opt, d_norm] = getRandomPolicy(game);

A = eye(game.N_states);
B = [1; 0];
mult1 = kron(A,B);
B = [0; 1];
mult2 = kron(A,B);

% % % % AGENTE:
% numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
% numRep = 2000; % Número de repeticiones de cada set de episodios
% numEpisodes = 1; % Número de episodios de cada repeticion
maxNumStepsPerEpisode = 500; % Número máximo de pasos en cada episodio
G = zeros(numExperiments, numRep*numEpisodes); % Reward por episodio
% epsilon = 0; % e-greedy value (entre 0.05 y 0.1)
% alphaD = 0.01; % Stepsize para la iteración de la variable dual d
% alphaTD = 0.11; % Stepsize para la iteración de la variable primal v

S = game.N_states; % Número de estados
A = game.N_actions; % Número de acciones
mu = game.mu; % Distribución inicial de probabilida de los estados
P = game.P; % Matriz de transiciones
R = game.R; % Vector de rewards
terminal = false; % flag used when terminal state reached

% Optimum values
v_opt = inv(eye(game.N_states)-game.gamma*game.pi_opt*game.P)*game.pi_opt*game.R;
q_opt = inv(eye(game.N_states*game.N_actions)-game.gamma*game.P*game.pi_opt)*game.R;
% Valor óptimo de d (política en forma vector)
d_opt_norm = getPolicyVector(q_opt, game); % policy óptima

% Variable que acumulará la funcion V y el error en la política al final de cada episodio
Vs_acumulada = nan(game.N_states, numRep*numEpisodes, numExperiments);
errorD = nan(numExperiments, numRep*numEpisodes);
d_norm_acumulada = nan(game.N_states*game.N_actions, numRep*numEpisodes, numExperiments);
% d_acumulada = nan(game.N_states*game.N_actions, numRep*numEpisodes, numExperiments);

for exp = 1:numExperiments
    not_error = true;
    % Inicializamos D y V
    d = rand(S*A,1);
    v = rand(S,1);
    v(1) = 0; v(end) = 0;
%     exp
    episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
    episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
    for k = 1:numRep
        s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
        totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
        
        for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
            % currentState = game.centralState; % estado inicial el central
            % currentState = randi([2 game.centralState]);
            currentState = 2; % empezar en el de la izquierda
            terminal = false; % true when episode finish, false otherwise
            stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
            
            % Normalize d
            d_norm = getPolicyVectorFromD( d, game );
            % Get policy matrix
            policy_by_action = reshape(d_norm', [A,S])';
            policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';
            while ~terminal && not_error %&& sum(d((((currentState-1)*A)+1):currentState*A)) ~= 0
                % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
                currentAction = e_greedy(d_norm, epsilon, currentState, game.N_actions);
                
                % Tomamos la acción a (currentAction), observamos la recompensa
                % r (reward) y el siguiente estado s' (nextState).
                [nextState, ~, reward] = getNextState(game, currentState, currentAction);
                G(exp, (k-1)*numEpisodes+n) = (game.gamma^stepPerEpisode)*reward+G(exp,(k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
                
                % Update de v(s)
                % TEMPORAL DIFFERENCE
                v(currentState) = v(currentState) + alphaTD*(reward + game.gamma*v(nextState) - v(currentState)); % policy evaluation
                % EXACTA (BELLMAN)
                % v = (inv(eye(S)-game.gamma*policy_matrix*P))*policy_matrix*R;
                
                % Almacenamos las transiciones del episodio
                s_a_sNext(totalStepsPerRep,:) = [currentState currentAction nextState reward stepPerEpisode];
                
                % Actualizamos valores
                stepPerEpisode = stepPerEpisode + 1;
                totalStepsPerRep = totalStepsPerRep + 1;
                currentState = nextState;
                
                % Evaluación de si el episodio ha terminado o no
                if sum(currentState == game.finalState) == 1 || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
                    terminal = true;
                    episodeCountV = episodeCountV + 1;
                    Vs_acumulada(:, episodeCountV, exp) = v;
                end
            end
        end
        totalStepsPerRep = totalStepsPerRep-1; % Compensamos el que se incrementó de más
        
        for i = 1:totalStepsPerRep
            % Recover saved episodes
            [currentState, nextState, reward, stepPerEpisode, s_a_index] = recoverSavedEpisode(s_a_sNext, game.N_actions, i);
            
            % Policy (or d) update
            d(s_a_index) = d(s_a_index) + alphaD*(reward + game.gamma*v(nextState) - v(currentState));
            d_orig = d;
            d(d<0)=0; % Projection of d over positives
            
            % Normalize d
            d_norm = getPolicyVectorFromD( d, game );
            
            % Evaluación de si el episodio ha terminado o no para guardar el error en la policy (save policy error)
            if any(isnan(d_norm))
                % Fix de la d original (que podía tener números negativos)
                d_orig(isnan(d_norm) & d_orig<0) = abs(d_orig(isnan(d_norm) & d_orig<0));
                d = d_orig;
                d(d<0)=0; % Projection of d over positives
                d_norm = getPolicyVectorFromD( d, game );
                % disp('nan!')
                % not_error = false;
                % break;
            end
            if sum(nextState == game.finalState) == 1 || stepPerEpisode == maxNumStepsPerEpisode-1 % Si el estado siguiente es el terminal
                episodeCountD = episodeCountD + 1;
                % Calculate norm-2 of policy error
                errorD(exp, episodeCountD) = norm(abs(d_norm(game.N_actions+1:end-game.N_actions) - d_opt_norm(game.N_actions+1:end-game.N_actions)),2);
                d_norm_acumulada(:,episodeCountD, exp) = d_norm;
                d_acumulada(:,episodeCountD, exp) = d_orig;
                % [d_norm; episodeCountD]
                % reshape(d_norm', [2 21])'
            end
        end
    end
end

% % Apaño para cuando ha aparecido un NaN. Replicamos el último valor bueno
% % para el resto de episodios
% for i = 1:size(errorD,1)
%     ind_nan = find(isnan(errorD(i,:))==1);
%     if ~isempty(ind_nan)
%         errorD(i,ind_nan) = errorD(i,ind_nan(1)-1);
%     end
% end
Vs_mean = mean(squeeze(Vs_acumulada(:,end,:)),2);
[Qsa_mean, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, game);

% % % % REPRESENTACIÓN DE RESULTADOS:
% % Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
% figure, plot((1:numRep*numEpisodes), G, 'LineWidth', 2)
% title(['Reward per episode (always starting at s=' ,num2str(game.centralState), ')'])
% xlabel('Episode'), ylabel('G')
%
% aux1 = sum(Qsa_acumulada);
% aux2 = squeeze(aux1)';
% sumQ_mean = mean(aux2);
% figure, plot((1:numRep*numEpisodes), sumQ_mean, 'LineWidth', 2)
% title('\Sigmaq(s,a) for each episode')
% xlabel('Episode'), ylabel('\Sigmaq(s,a)')
%
% aux1 = sum(Vs_acumulada);
% aux2 = squeeze(aux1)';
% sumV_mean = mean(aux2);
% figure, plot((1:numRep*numEpisodes), sumV_mean, 'LineWidth', 2)
% title('\Sigmav(s) for each episode')
% xlabel('Episode'), ylabel('\Sigmav(s)')
%
% aux1 = sum((Qsa_acumulada(game.N_actions+1:end-game.N_actions,:,:)-q_opt(game.N_actions+1:end-game.N_actions,:)).^2 , 1);
% aux2 = squeeze(aux1)';
% mse_q = sqrt(mean(aux2));
% figure, plot((1:numRep*numEpisodes), mse_q, 'LineWidth', 2)
% title('Mean-squared error of q(s,a) for each episode')
% xlabel('Episode'), ylabel('MSE q(s,a)')
%
% aux1 = sum((Vs_acumulada(2:end-1,:,:)-v_opt(2:end-1,:)).^2 , 1);
% aux2 = squeeze(aux1)';
% mse_v = sqrt(mean(aux2));
% figure, plot((1:numRep*numEpisodes), mse_v, 'LineWidth', 2)
% title('Mean-squared error of v(s) for each episode')
% xlabel('Episode'), ylabel('MSE v(s)')

errorD_mean = mean(errorD,1); % Media de los experimentos realizados
% figure, plot((1:numRep*numEpisodes), errorD_mean, 'LineWidth', 2)
% title('Policy error')
% xlabel('Episode'), ylabel('d-d_{opt}')
save(['Resultados exacto\numExperiments=' num2str(numExperiments) ',numRep=' num2str(numRep) ',numEpisodes=' num2str(numEpisodes) ',alphaD=' num2str(alphaD) ',alphaTD=' num2str(alphaTD) ',epsilon=' num2str(epsilon) '.mat'])
end