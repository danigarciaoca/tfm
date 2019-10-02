% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Nombre del fichero: double_q_learning.m                           %
% Autor: Daniel M. García-Ocaña Hernández                           %
% Fecha de creación: 08/05/2017                                     %
% Implementación de Double Q-learning                               %
% (p.137 de Reinforcement Learning: An Introduction)                %
% Ejemplo: Cliff Walking, p.138 Example 6.6                         %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

clear all, close all, clc

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

load('policy_matrix_qlearning.mat'), pi_opt = policy_matrix;
% Optimum value
v_opt=inv(eye(S)-gamma*pi_opt*env.P)*pi_opt*env.R;
q_opt=inv(eye(S*A)-gamma*env.P*pi_opt)*env.R;
q_opt_format = reshape(q_opt, [A S])';

% % % % AGENTE:
numExperiments = 100; % Número de experimentos sobre el que se promedia
numEpisodes = 500; % Número de episodios de cada experimento
max_steps = 10000;
G = zeros(numExperiments,numEpisodes); % Return por episodio
G_eps0_acumulada = zeros(numExperiments,numEpisodes); % Return por episodio cuando epsilon = 0
epsilon = 0.001;
alfa = 0.95;

% Variable que acumulará las funciones Q y V al final de cada episodio
Qsa_lineal_acumulada1 = zeros(S*A, numEpisodes, numExperiments);
Qsa_lineal_acumulada2 = zeros(S*A, numEpisodes, numExperiments);
Vs_acumulada1 = zeros(S, numEpisodes, numExperiments);
Vs_acumulada2 = zeros(S, numEpisodes, numExperiments);

for k = 1:numExperiments
    % Inicializamos Q para cada experimento
    terminal_states = [env.cliff env.finalState];
    [Qsa1, Qsa2] = initializeDoubleQfunction(S, A, terminal_states);
    k
    for i = 1:numEpisodes
        % Initialize episode
        terminal = false;
        step = 0;
        s_t = env.initState; %env.initState = initPos( 'random', env.board, S ); % Comentar esta línea si queremos empezar desde la esquina (es el que viene por defecto)
        
        while ~terminal % siempre que el episodio no haya terminado
            % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
            Qsa = (Qsa1+Qsa2)./2;
            a_t = e_greedy(Qsa, epsilon, s_t, A);
            
            % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
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
        % Acumulamos el valor de la función V(s)
        [ ~, policy_matrix] = getPolicy(Qsa1, env);
        Vs_acumulada1(:, i, k) = getValueFunction(Qsa1, policy_matrix, env);
        [ ~, policy_matrix] = getPolicy(Qsa2, env);
        Vs_acumulada2(:, i, k) = getValueFunction(Qsa2, policy_matrix, env);
        
        % Acumulamos el valor de la función Q(s,a)
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

Qsa_and_policy_opt1.Qsa(37,:)
Qsa_and_policy_opt2.Qsa(37,:)
q_opt_format(37,:)

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
figure, hold on
plot((1:numEpisodes), Gmean, 'b', 'LineWidth', 2)
plot((1:numEpisodes), Gmean_eps0, 'g', 'LineWidth', 2)
plot((1:numEpisodes), mean(Gmean_eps0)*ones(size(Gmean_eps0)),'--r', 'LineWidth', 2)
hold off, title(['Return per episode (always starting at s=' ,num2str(env.initState), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon)], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0']},'Location','southeast','FontSize',12)

checkCliffResults(env, q_opt_format, Qsa_and_policy_opt1)