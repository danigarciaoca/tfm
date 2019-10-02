% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Nombre del fichero: q_learning.m                                  %
% Autor: Daniel M. Garc�a-Oca�a Hern�ndez                           %
% Fecha de creaci�n: 28/11/2016                                     %
% Fecha de actualizaci�n - v1.0: 08/05/2017                         %
% Implementaci�n de Q-learning: an off-policy TD control algorithm  %
% (p.137 de Reinforcement Learning: An Introduction)                %
% Ejemplo: Cliff Walking, p.138 Example 6.6                         %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

clear all, clc

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

% Optimum value (value iteration)
N_steps = 1000;
[v_opt, q_opt, q_opt_format] = value_iteration(env, N_steps);
[~, pi_lineal] = max(q_opt_format,[],2);
grid_result = gridPolicyRepresentation(env, pi_lineal)
[num, ind] = max(q_opt_format(37,:))

% % % % AGENTE:
numExperiments = 50; % N�mero de experimentos sobre el que se promedia
numEpisodes = 2000; % N�mero de episodios de cada experimento
max_steps = 10000;
G = zeros(numExperiments,numEpisodes); % Return por episodio
G_eps0_final = zeros(numExperiments,1); % Return por episodio cuando epsilon = 0
G_eps0_each_epi = zeros(numExperiments, numEpisodes); % Return por episodio cuando epsilon = 0
epsilon = 0.0005;
alfa = 0.2;

% Variable que acumular� las funciones Q y V al final de cada episodio
Qsa_lineal_acumulada = zeros(S*A, numEpisodes, numExperiments);
Vs_acumulada = zeros(S, numEpisodes, numExperiments);

for k = 1:numExperiments
    % Inicializamos Q para cada experimento
    Qsa = rand(S, A);
    Qsa(env.finalState,:) = zeros(size(Qsa(env.finalState,:)));
    Qsa(env.cliff,:) = zeros(size(Qsa(env.cliff,:)));
    k
    for i = 1:numEpisodes
        % Initialize episode
        terminal = false;
        step = 0;
        s_t = env.initState; %env.initState = initPos( 'random', env.board, S ); % Comentar esta l�nea si queremos empezar desde la esquina (es el que viene por defecto)
        
        while ~terminal % siempre que el episodio no haya terminado
            % Escogemos A (currentAction) de S (currentState) seg�n la e-greedy policy.
            a_t = e_greedy(Qsa, epsilon, s_t, A);
            
            % Tomamos la acci�n 'a', observamos la recompensa 'r' y el siguiente estado s'.
            [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
            G(k,i) = G(k,i) + gamma^step * reward;
            
            % Actualizamos el valor de Q(s,a)
            a_t1 = greedy(Qsa, s_t1);
            Qsa(s_t, a_t) = Qsa(s_t, a_t) + alfa*(reward + gamma*Qsa(s_t1, a_t1) - Qsa(s_t, a_t));
            
            % Actualizamos los valores
            s_t = s_t1;
            step = step +1;
            
            % Terminate the episode
            if step == max_steps || terminal
                [~, ~, G_eps0] = q_learning_core(env, Qsa, 0, alfa, numEpisodes, max_steps);
                G_eps0_each_epi(k,i) = G_eps0;
                break
            end
        end
        % Acumulamos el valor de la funci�n V(s)
        [ ~, policy_matrix] = getPolicy(Qsa, env);
        Vs_acumulada(:, i, k) = getValueFunction(Qsa, policy_matrix, env);
        
        % Acumulamos el valor de la funci�n Q(s,a)
        Qsa_lineal_acumulada(:, i, k) = reshape(Qsa', [S*A 1]);
    end
    % Once the policy/Q(s,a) has converged, we evaluate the problem with
    % epsilon = 0 (i.e, greedy policy)
    [~, ~, G_eps0] = q_learning_core(env, Qsa, 0, alfa, numEpisodes, max_steps);
    G_eps0_final(k,:) = G_eps0;
end

% Calculamos la media de todas las Q(s,a) acumuladas en todos los experimentos
[Qsa_and_policy_opt, policy_matrix] = getPolicy(Qsa_lineal_acumulada, env);
Vs = getValueFunction(Qsa_and_policy_opt.Qsa, policy_matrix, env);

% % % REPRESENTACI�N DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0_each_epi = mean(G_eps0_each_epi,1); % Media de los experimentos realizados cuando epsilon = 0 (evoluci�n)
Gmean_eps0 = mean(G_eps0_final,1); % Media de los experimentos realizados cuando epsilon = 0 (ya cnvergido)
figure, hold on
plot((1:numEpisodes), Gmean, 'b', 'LineWidth', 2)
plot((1:numEpisodes), Gmean_eps0_each_epi, 'k', 'LineWidth', 2)
% plot((1:numEpisodes), Gmean_eps0, 'g', 'LineWidth', 2)
plot((1:numEpisodes), mean(Gmean_eps0)*ones(size(Gmean_eps0)),'--r', 'LineWidth', 2)
hold off, title(['Q Return per episode (always starting at s=' ,num2str(env.initState), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon)], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0']},'Location','southeast','FontSize',12)

checkCliffResults(env, q_opt_format, Qsa_and_policy_opt)