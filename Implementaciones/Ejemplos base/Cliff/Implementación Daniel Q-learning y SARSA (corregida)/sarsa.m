% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Nombre del fichero: sarsa.m                                 %
% Autor: Daniel M. García-Ocaña Hernández                     %
% Fecha de creación: 28/11/2016                               %
% Fecha de actualización - v1.0: 08/05/2017                   %
% Implementación de SARSA: an on-policy TD control algorithm  %
% (p.135 de Reinforcement Learning: An Introduction)          %
% Ejemplo: Cliff Walking, p.138 Example 6.6                   %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

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

load('policy_matrix_sarsa.mat'), pi_opt = policy_matrix;
% Optimum value
v_opt=inv(eye(S)-gamma*pi_opt*env.P)*pi_opt*env.R;
q_opt=inv(eye(S*A)-gamma*env.P*pi_opt)*env.R;
q_opt_format = reshape(q_opt, [A S])';

% % % % AGENTE:
numExperiments = 50; % Número de experimentos sobre el que se promedia
numEpisodes = 500; % Número de episodios de cada experimento
max_steps = 10000;
G = zeros(numExperiments,numEpisodes); % Reward por episodio
epsilon = 0.1;
alfa = 0.1;

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
        % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
        a_t = e_greedy(Qsa, epsilon, s_t, env.numActions);
        
        while ~terminal % siempre que el episodio no haya terminado
            
            % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
            [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
            G(k,i) = G(k,i) + gamma^step * reward;
            
            % Escogemos A' (nextAction) de S' (nextState) según la e-greedy policy.
            a_t1 = e_greedy(Qsa, epsilon, s_t1, env.numActions);
            
            % Actualizamos el valor de Q(s,a)
            Qsa(s_t, a_t) = Qsa(s_t, a_t) + alfa*(reward + gamma*Qsa(s_t1, a_t1) - Qsa(s_t, a_t));
            
            % Actualizamos los valores
            s_t = s_t1;
            a_t = a_t1;
            
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
end

% Calculamos la media de todas las Q(s,a) acumuladas en todos los experimentos
[Qsa_and_policy_opt, policy_matrix] = getPolicy(Qsa_lineal_acumulada, env);
Vs = getValueFunction(Qsa_and_policy_opt.Qsa, policy_matrix, env);

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
figure, plot((1:numEpisodes), Gmean, 'LineWidth', 2)
title(['Return per episode (always starting at s=' ,num2str(env.initState), ')'])
xlabel('Episode'), ylabel('G')

checkCliffResults(env, q_opt_format, Qsa_and_policy_opt)