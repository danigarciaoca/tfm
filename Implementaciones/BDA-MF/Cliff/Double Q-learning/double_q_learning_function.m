function double_q_learning_function(numExperiments, numEpisodes, max_steps, epsilon, alfa, rltv_path)
% % % % ENTORNO:
name = 'Cliff';
numFilTablero = 4;
numColTablero = 12;
rewards = [-1, -100]; % -1 en todas las transiciones, -100 si cae al cliff
initType = 'fixed'; % fixed: empieza en la esquina inferior izquierda; random: empieza en cualquier estado
env = GetCliffEnv(name, numFilTablero, numColTablero, rewards, initType);

% Asignaciones por comodidad
DoAction = env.DoAction;
S = env.numStates;
A = env.numActions;
gamma = env.gamma;

% % % % AGENTE:
G = zeros(numExperiments,numEpisodes); % Return por episodio
G_eps0_acumulada = zeros(numExperiments,numEpisodes); % Return por episodio cuando epsilon = 0

% Variable que acumular� las funciones Q y V al final de cada episodio
Qsa_lineal_acumulada1 = zeros(S*A, numEpisodes, numExperiments);
Qsa_lineal_acumulada2 = zeros(S*A, numEpisodes, numExperiments);
Vs_acumulada1 = zeros(S, numEpisodes, numExperiments);
Vs_acumulada2 = zeros(S, numEpisodes, numExperiments);

for k = 1:numExperiments
    % Inicializamos Q para cada experimento
    terminal_states = [env.cliff env.finalState];
    [Qsa1, Qsa2] = initializeDoubleQfunction(S, A, terminal_states);
    
    for i = 1:numEpisodes
        % Initialize episode
        terminal = false;
        step = 0;
        s_t = env.initState; %env.initState = initPos( 'random', env.board, S ); % Comentar esta l�nea si queremos empezar desde la esquina (es el que viene por defecto)
        
        while ~terminal % siempre que el episodio no haya terminado
            % Escogemos A (currentAction) de S (currentState) seg�n la e-greedy policy.
            Qsa = (Qsa1+Qsa2)./2;
            a_t = e_greedy(Qsa, epsilon, s_t, A);
            
            % Tomamos la acci�n 'a', observamos la recompensa 'r' y el siguiente estado s'.
            [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
            G(k,i) = G(k,i) + gamma^step * reward;
            
            % Actualizamos el valor de Q(s,a) de acuerdo a double Q-learning
            flip_coin = binornd(1,0.5); % Con probabilidad 0.5 hay exito (1); con probabilidad  1-0.5=0.5 hay fracaso (0)
            if flip_coin == 0
                a_t1 = greedy(Qsa1, s_t1);
                Qsa1(s_t, a_t) = Qsa1(s_t, a_t) + alfa*(reward + gamma*Qsa2(s_t1, a_t1) - Qsa1(s_t, a_t));
            elseif flip_coin == 1
                a_t1 = greedy(Qsa2, s_t1);
                Qsa2(s_t, a_t) = Qsa2(s_t, a_t) + alfa*(reward + gamma*Qsa1(s_t1, a_t1) - Qsa2(s_t, a_t));
            end
            
            % Actualizamos los valores
            s_t = s_t1;
            step = step + 1;
            
            % Terminate the episode
            if step == max_steps
                break
            end
        end
        % Acumulamos el valor de la funci�n V(s)
        [ ~, policy_matrix] = getPolicy(Qsa1, env);
        Vs_acumulada1(:, i, k) = getValueFunction(Qsa1, policy_matrix, env);
        [ ~, policy_matrix] = getPolicy(Qsa2, env);
        Vs_acumulada2(:, i, k) = getValueFunction(Qsa2, policy_matrix, env);
        
        % Acumulamos el valor de la funci�n Q(s,a)
        Qsa_lineal_acumulada1(:, i, k) = reshape(Qsa1', [S*A 1]);
        Qsa_lineal_acumulada2(:, i, k) = reshape(Qsa2', [S*A 1]);
    end
    % Once the policy/Q(s,a) has converged, we evaluate the problem with
    % epsilon = 0 (i.e, greedy policy)
    [~, ~, ~, ~, G_eps0] = double_q_learning_core(env, Qsa1, Qsa2, 0, alfa, numEpisodes, max_steps);
    G_eps0_acumulada(k,:) = G_eps0;
end

% Calculamos la media de todas las Q(s,a) acumuladas en todos los experimentos
[Qsa_and_policy_opt1, policy_matrix] = getPolicy(Qsa_lineal_acumulada1, env);
Vs1 = getValueFunction(Qsa_and_policy_opt1.Qsa, policy_matrix, env);
[Qsa_and_policy_opt2, policy_matrix] = getPolicy(Qsa_lineal_acumulada2, env);
Vs2 = getValueFunction(Qsa_and_policy_opt2.Qsa, policy_matrix, env);

% % % REPRESENTACI�N DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
mean_Gmean_eps0 = mean(Gmean_eps0); % Media de los episodios, promediados sobre los experimentos realizados, cuando epsilon = 0

save([rltv_path '\nExp=' num2str(numExperiments) ',nEpi=' num2str(numEpisodes) ',alpha=' num2str(alfa) ',eps=' num2str(epsilon) '.mat'])
end