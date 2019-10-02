function q_learning_function(numExperiments, numEpisodes, max_steps, epsilon, alfa, rltv_path)
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
S = env.numStates;
A = env.numActions;
gamma = env.gamma;

% % % % AGENTE:
G = zeros(numExperiments,numEpisodes); % Return por episodio
G_eps0_acumulada = zeros(numExperiments,numEpisodes); % Return por episodio cuando epsilon = 0

% Variable que acumulará las funciones Q y V al final de cada episodio
Qsa_lineal_acumulada = zeros(S*A, numEpisodes, numExperiments);
Vs_acumulada = zeros(S, numEpisodes, numExperiments);

for k = 1:numExperiments
    % Inicializamos Q para cada experimento
    Qsa = rand(S, A);
    Qsa(env.finalState,:) = zeros(size(Qsa(env.finalState,:)));
    Qsa(env.cliff,:) = zeros(size(Qsa(env.cliff,:)));
    
    for i = 1:numEpisodes
        % Initialize episode
        terminal = false;
        step = 0;
        s_t = env.initState; %env.initState = initPos( 'random', env.board, S ); % Comentar esta línea si queremos empezar desde la esquina (es el que viene por defecto)
        
        while ~terminal % siempre que el episodio no haya terminado
            % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
            a_t = e_greedy(Qsa, epsilon, s_t, A);
            
            % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
            [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
            G(k,i) = G(k,i) + gamma^step * reward;
            
            % Actualizamos el valor de Q(s,a)
            a_t1 = greedy(Qsa, s_t1);
            Qsa(s_t, a_t) = Qsa(s_t, a_t) + alfa*(reward + gamma*Qsa(s_t1, a_t1) - Qsa(s_t, a_t));
            
            % Actualizamos los valores
            s_t = s_t1;
            step = step +1;
            
            % Terminate the episode
            if step == max_steps
                break
            end
        end
        % Acumulamos el valor de la función V(s)
        [ ~, policy_matrix] = getPolicy(Qsa, env);
        Vs_acumulada(:, i, k) = getValueFunction(Qsa, policy_matrix, env);
        
        % Acumulamos el valor de la función Q(s,a)
        Qsa_lineal_acumulada(:, i, k) = reshape(Qsa', [S*A 1]);
    end
    % Once the policy/Q(s,a) has converged, we evaluate the problem with
    % epsilon = 0 (i.e, greedy policy)
    [~, ~, G_eps0] = q_learning_core(env, Qsa, 0, alfa, numEpisodes, max_steps);
    G_eps0_acumulada(k,:) = G_eps0;
end

% Calculamos la media de todas las Q(s,a) acumuladas en todos los experimentos
[Qsa_and_policy_opt, policy_matrix] = getPolicy(Qsa_lineal_acumulada, env);
Vs = getValueFunction(Qsa_and_policy_opt.Qsa, policy_matrix, env);

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
mean_Gmean_eps0 = mean(Gmean_eps0); % Media de los episodios, promediados sobre los experimentos realizados, cuando epsilon = 0

save([rltv_path '\nExp=' num2str(numExperiments) ',nEpi=' num2str(numEpisodes) ',alpha=' num2str(alfa) ',eps=' num2str(epsilon) '.mat'])
end