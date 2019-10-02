clear all, clc
addpath('env_def')

% % % % ENVIRONMENT:
env = GetChainWalkEnv();

% Get problem data
S = env.S; % number of states
A = env.num_actions; % number of actions
N = env.N; % number of state features
Pssa = env.Pssa; % transition probability matrix
R = env.Rs; % state-rewards vector
gamma = env.gamma; % discount rate/factor
DoAction = env.DoAction;
GetStateFeatures = env.GetStateFeatures;

% Auxiliar matrixes to compute policy matrix
aux1 = eye(S);
aux2 = [1; 0];
mult1 = kron(aux1,aux2);
aux2 = [0; 1];
mult2 = kron(aux1,aux2);

% aux1 = eye(S);
% aux2 = eye(A);
% for i=1:A
%     mult(:,:,i) = kron(aux1,aux2(:,i));
% end
% policy_matrix2 = zeros(S,S*A);

% % % % AGENT:
numExperiments = 100; % Número de experimentos de numRep*numEpisodes episodios
numRep = 10; % Número de repeticiones de cada set de episodios
numEpisodes = 1; % Número de episodios de cada repeticion
maxStepsEpisode = 200; % Número máximo de pasos en cada episodio
epsilon = 0.2; %1 % e-greedy value (entre 0.05 y 0.2)
alphaD = 0.5; % Stepsize para la iteración de la variable dual d
G = zeros(numExperiments, numRep*numEpisodes); % Reward per episodio and experiment

% Optimum d vector (policy in vector form)
d_opt_norm = getPolicyVector(env.q_opt, env); % policy óptima


% Variable que acumulará la funcion V y el error en la política al final de cada episodio
Vs_acumulada = nan(S, numRep*numEpisodes, numExperiments);
errorD = nan(numExperiments, numRep*numEpisodes);
d_norm_acumulada = nan(S*A, numRep*numEpisodes, numExperiments);

phi = GetFeatureMatrix(S, N, GetStateFeatures, env);

for exp = 1:numExperiments
    % Initialize vector 'd' representing random initial policy
    d = rand(S*A,1); % d >= 0
    % d = d / sum(d);  % sum(d) = 1
    % d = [0.3151    0.2637    0.1879    0.0220    0.0083    0.0556    0.0946    0.0528]';
    exp
    
    episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
    episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
    

    for k = 1:numRep
        s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
        totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
        phi_t = nan(numEpisodes*maxStepsEpisode, N);
        phi_t1 = nan(numEpisodes*maxStepsEpisode, N);
        reward_t = nan(numEpisodes*maxStepsEpisode, 1);
        
        for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
            % currentState = game.centralState; % estado inicial el central
            % currentState = randi([2 game.centralState]);
            s = env.initial_state; % empezar en el de la izquierda
            terminal = false; % true when episode finish, false otherwise
            stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
            
            % Normalize d
            d_norm = getPolicyVectorFromD( d, env );
            % Get policy matrix
            policy_by_action = reshape(d_norm', [A,S])';
            policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';
%             for i=1:A
%                 policy_matrix2 = policy_matrix2 + diag(policy_by_action(:,i))*mult(:,:,i)';
%             end
            
            while true
                % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
                a = e_greedy(d_norm, epsilon, s, A);
                
                % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
                [s_t1, r, terminal] = DoAction( a, s, env );
                %[s_t1, r] = getNextState(env, currentState, currentAction);
                G(exp, (k-1)*numEpisodes+n) = (gamma^stepPerEpisode)*r+G(exp,(k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
                
                % Update de theta
                % LEAST-SQUARES TEMPORAL DIFFERENCE (I)
                phi_t(totalStepsPerRep,:) = GetStateFeatures(s);
                phi_t1(totalStepsPerRep,:) = GetStateFeatures(s_t1);
                reward_t(totalStepsPerRep,:) = r; 

                % EXACTA (MEDIANTE APROXIMACIÓN LINEAL)
                % theta = GetExactParamOpt( policy_matrix*R, policy_matrix*P, phi, gamma);
                % v = phi*theta;
                
                % Almacenamos las transiciones del episodio
                s_a_sNext(totalStepsPerRep,:) = [s a s_t1 r stepPerEpisode];
                
                % Actualizamos valores
                stepPerEpisode = stepPerEpisode + 1;
                totalStepsPerRep = totalStepsPerRep + 1;
                s = s_t1;
                
                % Evaluate if episode has finished or not
                if (stepPerEpisode == maxStepsEpisode) || (terminal == true) % if maximum numbero of steps per episode or terminal state reached
                    episodeCountV = episodeCountV + 1;
                    % Vs_acumulada(:, episodeCountV, exp) = v;
                    % disp(['Fin' num2str(n) ' y ' num2str(stepPerEpisode)])
                    break
                end
            end
        end

        totalStepsPerRep = totalStepsPerRep-1; % Compensamos el que se incrementó de más
        % LEAST-SQUARES TEMPORAL DIFFERENCE (II)
        theta = GetStochParam( reward_t, phi_t, phi_t1, gamma);
        % estimate value function
        v = phi*theta;
        
        % d_norm
        for i = 1:totalStepsPerRep
            % Recover saved episodes
            [s, s_t1, r, stepPerEpisode, s_a_index] = recoverSavedEpisode(s_a_sNext, A, i, env);
            
            % Policy (or d) update
            %d(s_a_index) = d(s_a_index) + alphaD*(reward + game.gamma*P(s_a_index,:)*phi*theta - phi(currentState,:)*theta);
            d(s_a_index) = d(s_a_index) + alphaD*(r + gamma*v(s_t1) - v(s));
            d_orig = d;
            d(d<0)=0; % Projection of d over positives
            
            % Normalize d
            d_norm = getPolicyVectorFromD(d, env);
            
            % Evaluación de si el episodio ha terminado o no para guardar el error en la policy (save policy error)
            if any(isnan(d_norm))
                % Fix de la d original (que podía tener números negativos)
                d_orig(isnan(d_norm) & d_orig<0) = abs(d_orig(isnan(d_norm) & d_orig<0));
                d = d_orig;
                d(d<0)=0; % Projection of d over positives
                d_norm = getPolicyVectorFromD( d, env );
            end

            if stepPerEpisode == maxStepsEpisode-1 % Si el estado siguiente es el terminal
                episodeCountD = episodeCountD + 1;
                % Calculate norm-2 of policy error
                errorD(exp, episodeCountD) = norm(abs(d_norm - d_opt_norm),2);
                d_norm_acumulada(:,episodeCountD, exp) = d_norm;
                d_acumulada(:,episodeCountD, exp) = d_orig;
                % d_norm'
                % [d_norm; episodeCountD]
                % reshape(d_norm', [2 21])'
            end
        end
    end
end

Vs_mean = mean(squeeze(Vs_acumulada(:,end,:)),2);
% [Qsa_mean, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, game);

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
figure, plot((1:numRep*numEpisodes), Gmean, 'LineWidth', 2)
title(['Reward per episode (always starting at s=' ,num2str(env.initial_state), ')'])
xlabel('Episode'), ylabel('G')
ylim([0 9])

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
figure(10), plot((1:numRep*numEpisodes), errorD_mean, 'LineWidth', 2)
title('Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')