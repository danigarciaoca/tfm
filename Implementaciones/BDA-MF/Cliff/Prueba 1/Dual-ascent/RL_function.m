function RL_function(numExperiments, numRep, numEpisodes, max_steps, epsilon, alphaD, alphaTD, rltv_path)

% % % % ENTORNO:
% name = 'Cliff';
% numFilTablero = 4;
% numColTablero = 12;
% rewards = [-1, -100]; % -1 en todas las transiciones, -100 si cae al cliff
% initType = 'fixed'; % fixed: empieza en la esquina inferior izquierda; random: empieza en cualquier estado
% transitionType = 'rand';
% env = GetCliffEnv(name, numFilTablero, numColTablero, rewards, initType, transitionType);
load cliff_pruebas.mat

% Asignaciones por comodidad
DoAction = env.DoAction;
S = env.numStates; % N�mero de estados
A = env.numActions; % N�mero de acciones
env.mu(:) = (1/S)/(S-1); env.mu(env.initState) = 1-(1/S); mu = env.mu; % Distribuci�n inicial de probabilida de los estados
P = env.P; % Matriz de transiciones
R = env.R; % Vector de rewards
gamma = env.gamma;

% mult es un tensor auxiliar para crear la matriz de la pol�tica
aux1 = eye(S);
aux2 = eye(A);
for i=1:A
    mult(:,:,i) = kron(aux1,aux2(:,i));
end

G = zeros(numExperiments, numRep*numEpisodes); % Return por episodio
G_eps0_acumulada = zeros(numExperiments, numRep*numEpisodes); % Return por episodio cuando epsilon = 0

% Variable que acumular� la funcion V y el error en la pol�tica al final de cada episodio
Vs_acumulada = nan(S, numRep*numEpisodes, numExperiments);

% load('v_opt_q-learning.mat')
for exp = 1:numExperiments
    % Inicializamos D y V
    d = rand(S*A,1);
    v = rand(S,1);
    terminal_states = [env.cliff env.finalState];
    v(terminal_states) = 0;
    % v = Vs;
    
    exp
    
    episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
    episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
    for k = 1:numRep
        
        s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
        totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
        
        for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
            s_t = env.initState;
            terminal = false; % true when episode finish, false otherwise
            step = 0; % step counts the number of steps taken in ONE of the numEpi episodes simulated
            
            % Normalize d
            d_norm = getPolicyVectorFromD( d, env );
            
            % Get policy matrix
            policy_matrix = zeros(S,S*A);
            policy_by_action = reshape(d_norm', [A,S])';
            for i=1:A
                policy_matrix = policy_matrix + diag(policy_by_action(:,i))*mult(:,:,i)';
            end
            while ~terminal
                % Escogemos A (currentAction) de S (currentState) seg�n la e-greedy policy.
                a_t = e_greedy(d_norm, epsilon, s_t, A);
                
                % Tomamos la acci�n a (currentAction), observamos la recompensa
                % r (reward) y el siguiente estado s' (nextState).
                [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
                
                G(exp, (k-1)*numEpisodes+n) = gamma^step * reward + G(exp,(k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habr�a que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
                
                % Update de v(s)
                % TEMPORAL DIFFERENCE
                v(s_t) = v(s_t) + alphaTD*(reward + gamma*v(s_t1) - v(s_t)); % policy evaluation
                % EXACTA (BELLMAN)
                %v = (inv(eye(S)-gamma*policy_matrix*P))*policy_matrix*R;

                % Almacenamos las transiciones del episodio
                s_a_sNext(totalStepsPerRep,:) = [s_t a_t s_t1 reward step terminal];
                
                % Actualizamos valores
                step = step + 1;
                totalStepsPerRep = totalStepsPerRep + 1;
                s_t = s_t1;
                
                % Evaluaci�n de si el episodio ha terminado o no
                if terminal || step == max_steps % Si el estado actual es el terminal
                    episodeCountV = episodeCountV + 1;
                    Vs_acumulada(:, episodeCountV, exp) = v;
                    % disp(['Fin' num2str(n) ' y ' num2str(stepPerEpisode)])
                    break;
                end
            end
        end
        totalStepsPerRep = totalStepsPerRep-1; % Compensamos el que se increment� de m�s
        
        for i = 1:totalStepsPerRep
            % Recover saved episodes
            [s_t, s_t1, reward, step, s_a_index, terminal] = recoverSavedEpisode(s_a_sNext, A, i);
            
            % Policy (or d) update
            d(s_a_index) = d(s_a_index) + alphaD*(reward + gamma*v(s_t1) - v(s_t));
            d_orig = d;
            d(d<0)=0; % Projection of d over positives
            
            % Normalize d
            d_norm = getPolicyVectorFromD( d, env );
            
            % Evaluaci�n de si el episodio ha terminado o no para guardar el error en la policy (save policy error)
            if any(isnan(d_norm))
                % Fix de la d original (que pod�a tener n�meros negativos)
                d_orig(isnan(d_norm) & d_orig<0) = abs(d_orig(isnan(d_norm) & d_orig<0));
                d = d_orig;
                d(d<0)=0; % Projection of d over positives
                d_norm = getPolicyVectorFromD( d, env );
                % disp('nan!')
                % not_error = false;
                % break;
            end
            if terminal || step == max_steps-1 % Si el estado siguiente es el terminal
                episodeCountD = episodeCountD + 1;
                % Calculate norm-2 of policy error
                % errorD(exp, episodeCountD) = norm(abs(d_norm(A+1:end-A) - d_opt_norm(A+1:end-A)),2);
                % d_norm_acumulada(:,episodeCountD, exp) = d_norm;
                % d_acumulada(:,episodeCountD, exp) = d_orig;
                % [d_norm; episodeCountD]
                % reshape(d_norm', [2 21])'
                % [~, policy_lineal] = max(reshape(d_norm,[A S])',[],2);
                % checkCliffResults(env, policy_lineal)
            end
        end
    end
    [~, ~, G_eps0] = RL_core(env, d, v, 0, alphaTD, alphaD, numRep, numEpisodes, max_steps, mult);
    G_eps0_acumulada(exp,:) = G_eps0;
end

Vs_mean = mean(squeeze(Vs_acumulada(:,end,:)),2);
% [Qsa_mean, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, env);

% % % REPRESENTACI�N DE RESULTADOS:
% Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
mean_Gmean_eps0 = mean(Gmean_eps0); % Media de los episodios, promediados sobre los experimentos realizados, cuando epsilon = 0

save([rltv_path '\nExp=' num2str(numExperiments) ',nRep=' num2str(numRep) ',nEpi=' num2str(numEpisodes) ',alphaD=' num2str(alphaD) ',alphaTD=' num2str(alphaTD) ',eps=' num2str(epsilon) '.mat'])
end

