clear all, clc
addpath('env_def','buffer')

% % % % ENVIRONMENT:
N = 6;
features = 'RBF';
env = GetMountainCarEnv(features, N)

% Get problem data
S = env.S; % number of states
A = env.num_actions; % number of actions
N = env.N; % number of state features
% % % % % Pssa = env.Pssa; % transition probability matrix
% % % % % R = env.Rs; % state-rewards vector
gamma = env.gamma; % discount rate/factor
DoAction = env.DoAction;
GetStateFeatures = env.GetStateFeatures;
PlotEpisode = env.PlotEpisode;

% Auxiliar matrixes to compute policy matrix
aux1 = eye(S);
aux2 = [1; 0; 0];
mult1 = kron(aux1,aux2);
aux2 = [0; 1; 0];
mult2 = kron(aux1,aux2);
aux2 = [0; 0; 1];
mult3 = kron(aux1,aux2);


% Auxiliar matrixes to compute policy matrix
aux1 = eye(S);
aux2 = eye(A);
for i=1:A
    mult(:,:,i) = kron(aux1,aux2(:,i));
end
policy_matrix2 = zeros(S,S*A);

% % % % AGENT:
numExperiments = 20; % Número de experimentos de numRep*numEpisodes episodios
numRep = 100; % Número de repeticiones de cada set de episodios
numEpisodes = 1; % Número de episodios de cada repeticion
maxStepsEpisode = 500; % Número máximo de pasos en cada episodio
epsilon = 0.1; % e-greedy value (entre 0.05 y 0.2)
alphaD = 0.001; % Stepsize para la iteración de la variable dual d
G = zeros(numExperiments, numRep*numEpisodes); % Reward per episodio and experiment
G_eps0_acumulada = zeros(numExperiments, numRep*numEpisodes); % Return por episodio cuando epsilon = 0

% % % phi = GetFeatureMatrix(S, N, GetStateFeatures);
BUFF_DIM = numEpisodes*maxStepsEpisode;
phi_t = CreateCircularBuffer(BUFF_DIM, N^env.num_dim); % N columnas para los N parámetros a guardar
phi_t1 = CreateCircularBuffer(BUFF_DIM, N^env.num_dim); % N columnas para los N parámetros a guardar
reward_t = CreateCircularBuffer(BUFF_DIM, 1); % 1 columnas para el parámetro a guardar

for exp = 1:numExperiments
    % Initialize vector 'd' representing random initial policy
    d = rand(S*A,1); % d >= 0
    d = d / sum(d);  % sum(d) = 1
    exp
    
    % Inicializamos theta (no es necesario en realidad por cómo se calcula)
    theta = rand(N^2,1);
    
    % Inicializamos el buffer circular
    phi_t = RestoreBuffer(phi_t);
    phi_t1 = RestoreBuffer(phi_t1);
    reward_t = RestoreBuffer(reward_t);
    
    episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
    episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
    for k = 1:numRep
        s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
        totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
        
        for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
            % currentState = game.centralState; % estado inicial el central
            % currentState = randi([2 game.centralState]);
            s_t = env.initial_state; % empezar en el de la izquierda
            terminal = false; % true when episode finish, false otherwise
            stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
            
            % Normalize d
            d_norm = getPolicyVectorFromD( d, env );
            
            while true
                % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
                s_t_disc = GetDiscretizedState(s_t, env.xy_disc);
                a = e_greedy(d_norm, epsilon, s_t_disc, A);
                %fprintf('Parametros')
                %d_norm((((s_t_disc-1)*A)+1):s_t_disc*A)'
                %[s_t_disc env.actions_list(a)]
                %PlotEpisode(s_t, a, stepPerEpisode)
                
                % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
                [s_t1, r, terminal] = DoAction( a, s_t, env );
                %[s_t1, r] = getNextState(env, currentState, currentAction);
                G(exp, (k-1)*numEpisodes+n) = (gamma^stepPerEpisode)*r+G(exp,(k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
                
                % Update de theta
                % LEAST-SQUARES TEMPORAL DIFFERENCE (I)
                phi_t = AddItem(phi_t, GetStateFeatures(s_t, env));
                phi_t1 = AddItem(phi_t1, GetStateFeatures(s_t1, env));
                reward_t = AddItem(reward_t, r);
                
                % EXACTA (MEDIANTE APROXIMACIÓN LINEAL)
                % theta = GetExactParamOpt( policy_matrix*R, policy_matrix*P, phi, gamma);
                % v = phi*theta;
                
                % Almacenamos las transiciones del episodio
                s_a_sNext(totalStepsPerRep,:) = [s_t a s_t1 r stepPerEpisode];
                
                % Actualizamos valores
                stepPerEpisode = stepPerEpisode + 1;
                totalStepsPerRep = totalStepsPerRep + 1;
                s_t = s_t1;
                
                % Evaluate if episode has finished or not
                if (stepPerEpisode == maxStepsEpisode) || (terminal == true) % if maximum numbero of steps per episode or terminal state reached
                    episodeCountV = episodeCountV + 1;
                    % return al final del episodio
                    % Vs_acumulada(:, episodeCountV, exp) = v;
                    % disp(['Fin' num2str(n) ' y ' num2str(stepPerEpisode)])
                    break;
                end
            end
            fprintf('repetition: %d, episode: %d, reward: %.3f\n', k, n, G(exp, (k-1)*numEpisodes+n))
        end
        totalStepsPerRep = totalStepsPerRep-1; % Compensamos el que se incrementó de más
        
        if sum(sum(phi_t.array(phi_t.index:end,:))) == 0 % Si aún no se ha llenado el buffer
            real_buffer = phi_t.index-1;
            phi_t_data = phi_t.array(1:real_buffer,:);
            phi_t1_data = phi_t1.array(1:real_buffer,:);
            reward_t_data = reward_t.array(1:real_buffer,:);
        else
            real_buffer = BUFF_DIM;
            phi_t_data = phi_t.array;
            phi_t1_data = phi_t1.array;
            reward_t_data = reward_t.array;
        end
        % LEAST-SQUARES TEMPORAL DIFFERENCE (II)
        theta = GetStochParam( reward_t_data, phi_t_data, phi_t1_data, gamma);
        % % %         % estimate value function
        % % %         v = phi*theta;
        
        % d_norm
        for i = 1:real_buffer
            % Recover saved episodes
            [s_t, s_t1, r, stepPerEpisode, s_a_index] = recoverSavedEpisode(s_a_sNext, A, i, env);
            v_t = GetStateFeatures(s_t, env)'*theta;
            v_t1 = GetStateFeatures(s_t1, env)'*theta;
            
            % Policy (or d) update
            %d(s_a_index) = d(s_a_index) + alphaD*(reward + game.gamma*P(s_a_index,:)*phi*theta - phi(currentState,:)*theta);
            d(s_a_index) = d(s_a_index) + alphaD*(r + gamma*v_t1 - v_t);
            d_orig = d;
            d(d<0)=0; % Projection of d over positives
            d = d / sum(d);
            
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
            
            if stepPerEpisode == maxStepsEpisode-1 || r == 0 % Si el estado siguiente es el terminal
                episodeCountD = episodeCountD + 1;
                break;
            end
        end
    end
    G_eps0 = RL_mount_buffer_core(env, d, theta, 0, alphaD, numRep, numEpisodes, maxStepsEpisode, mult);
    G_eps0_acumulada(exp,:) = G_eps0;
end

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
figure, hold on
plot((1:numRep*numEpisodes), Gmean, 'b', 'LineWidth', 2)
plot((1:numRep*numEpisodes), Gmean_eps0, 'g', 'LineWidth', 2)
plot((1:numRep*numEpisodes), mean(Gmean_eps0)*ones(size(Gmean_eps0)),'--r', 'LineWidth', 2)
hold off, title(['Return per episode (always starting at s=' ,num2str(env.initial_state), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon)], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0']},'Location','southeast','FontSize',12)