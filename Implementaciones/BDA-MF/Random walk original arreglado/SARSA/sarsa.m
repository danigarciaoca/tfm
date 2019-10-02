clear all, clc

% % % % ENTORNO:
name = 'RandomWalk';
N_states = 13;
initType = 'leftCorner'; % {'center', 'leftCorner'}
transitionType = 'det'; % {'det', 'rand'}
rewardType = 'det'; % {'det', 'rand'} || if rand, extra argument: finalReward=50
env = GetRndWalkEnv(name, N_states, initType, transitionType, rewardType);

% Optimum values
% Optimum value (value iteration)
N_steps = 1000;
[v_opt, q_opt, q_opt_format] = value_iteration(env, N_steps, env.terminal_states);
[~, pi_lineal] = max(q_opt_format,[],2); % Comprobación policy obtenida

DoAction = env.DoAction;
S = env.numStates; % Número de estados
A = env.numActions; % Número de acciones

% % % % AGENTE:
numExperiments = 100; % Número de experimentos sobre el que se promedia
numEpisodes = 1000; % Número de episodios de cada experimento
maxNumStepsPerEpisode = 500; % Número máximo de pasos en cada episodio
G = zeros(numExperiments,numEpisodes); % Return por episodio
G_eps0_final = zeros(numExperiments, 1); % Return por episodio cuando epsilon = 0
G_eps0_each_epi = zeros(numExperiments, numEpisodes); % Return por episodio cuando epsilon = 0
epsilon = 0.15; % e-greedy value
alpha = 0.7;

% Variable que acumulará las funciones Q y V al final de cada episodio
Qsa_lineal_acumulada = zeros(S*A, numEpisodes, numExperiments);
Vs_acumulada = zeros(S, numEpisodes, numExperiments);

for exp = 1:numExperiments
    % Inicializamos Q para cada experimento
    % Qsa = zeros(S, A);
    % Inicializamos Q para cada experimento
    Qsa = rand(S, A);
    Qsa(1,:) = zeros(size(Qsa(1,:)));
    Qsa(end,:) = zeros(size(Qsa(1,:)));
    exp
    
    for i = 1:numEpisodes
        % Inicializamos S
        % currentState = game.centralState;
        s_t = env.initState; % empezar en el de la izquierda
        stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpisodes episodes simulated
        
        % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
        a_t = e_greedy(Qsa, epsilon, s_t, A);
        
        while true % siempre que el episodio no haya terminado
            
            % Tomamos la acción A (currentAction), observamos la recompensa R
            % (reward) y el siguiente estado S' (nextState).
            [ s_t1, reward, terminal ] = DoAction(a_t, s_t, env);
            
            % G(exp,i) = reward+game.gamma*G(exp,i); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
            G(exp,i) = (env.gamma^stepPerEpisode)*reward+G(exp,i);
            
            % Escogemos A' (nextAction) de S' (nextState) según la e-greedy policy.
            a_t1 = e_greedy(Qsa, epsilon, s_t1, A);
            
            % Actualizamos el valor de Q(s,a)
            Qsa(s_t, a_t) = Qsa(s_t, a_t) + alpha*(reward + env.gamma*Qsa(s_t1, a_t1) - Qsa(s_t, a_t));
            
            % Actualizamos los valores
            s_t = s_t1;
            a_t = a_t1;
            stepPerEpisode = stepPerEpisode + 1;
            
            % Evaluación de si el episodio ha terminado o no
            if terminal || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
                terminal = true;
                [~, ~, G_eps0] = sarsa_core(env, Qsa, 0, alpha, maxNumStepsPerEpisode);
                G_eps0_each_epi(exp,i) = G_eps0;
                break; % terminamos el episodio
            end
        end
        % Acumulamos el valor de la función Q(s,a
        Qsa_lineal_acumulada(:, i, exp) = reshape(Qsa', [S*A 1]);
        
        % Acumulamos el valor de la función V(s)
        policy = getPolicy(Qsa, env);
        Vs_acumulada(:, i, exp) = getValueFunction(Qsa, policy, env);
    end
    % Once the policy/Q(s,a) has converged, we evaluate the problem with
    % epsilon = 0 (i.e, greedy policy)
    [~, ~, G_eps0] = sarsa_core(env, Qsa, 0, alpha, maxNumStepsPerEpisode);
    G_eps0_final(exp,:) = G_eps0;
end

% getPolicy obtiene la policy del promedio de las Qsa de todos los
% experimentos realizados
policy = getPolicy(Qsa_lineal_acumulada, env);
Vs = getValueFunction(Qsa, policy, env);

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0_each_epi = mean(G_eps0_each_epi,1); % Media, de cada episodio, de los experimentos realizados con epsilon = 0
Gmean_eps0 = mean(G_eps0_final,1); % Media, tras convergencia, de los experimentos realizados cuando epsilon = 0
figure, hold on
plot((1:numEpisodes), Gmean, 'b', 'LineWidth', 2)
plot((1:numEpisodes), Gmean_eps0_each_epi, 'k', 'LineWidth', 2)
plot((1:numEpisodes), Gmean_eps0, 'g', 'LineWidth', 2)
plot((1:numEpisodes), mean(Gmean_eps0)*ones(numEpisodes,1),'--r', 'LineWidth', 2)
hold off, title(['Return per episode (always starting at s=' ,num2str(env.initState), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon)], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0']},'Location','southeast','FontSize',12)
xlim([0 500])

% aux1 = sum(Qsa_lineal_acumulada);
% aux2 = squeeze(aux1)';
% sumQ_mean = mean(aux2);
% figure, plot((1:numEpisodes), sumQ_mean, 'LineWidth', 2)
% title('\Sigmaq(s,a) for each episode')
% xlabel('Episode'), ylabel('\Sigmaq(s,a)')
%
% aux1 = sum(Vs_acumulada);
% aux2 = squeeze(aux1)';
% sumV_mean = mean(aux2);
% figure, plot((1:numEpisodes), sumV_mean, 'LineWidth', 2)
% title('\Sigmav(s) for each episode')
% xlabel('Episode'), ylabel('\Sigmav(s)')
%
% aux1 = sum((Qsa_lineal_acumulada(A+1:end-A,:,:)-q_opt(A+1:end-A,:)).^2 , 1);
% aux2 = squeeze(aux1)';
% mse_q = sqrt(mean(aux2));
% figure, plot((1:numEpisodes), mse_q, 'LineWidth', 2)
% title('Mean-squared error of q(s,a) for each episode')
% xlabel('Episode'), ylabel('MSE q(s,a)')
%
% aux1 = sum((Vs_acumulada(2:end-1,:,:)-v_opt(2:end-1,:)).^2 , 1);
% aux2 = squeeze(aux1)';
% mse_v = sqrt(mean(aux2));
% figure, plot((1:numEpisodes), mse_v, 'LineWidth', 2)
% title('Mean-squared error of v(s) for each episode')
% xlabel('Episode'), ylabel('MSE v(s)')

% Calcula el error en la policy en cada episodio, promediado entre todos
% los experimentos
policy_vector = getPolicyVector(Qsa_lineal_acumulada, env); % policy de cada experimento y para cada iteración
optimal_policy_vector = getPolicyVector(q_opt, env); % policy óptima
errorPolicy = policy_vector - optimal_policy_vector; % error de policies
dif = policy_vector(A+1:end-A,:,:) - optimal_policy_vector(A+1:end-A);
policy_error_mean = mean(sqrt(sum(dif.^2)),3); % ecm y promedio de entre todos los experimentos

figure, plot((1:numEpisodes), policy_error_mean, 'LineWidth', 2)
title('Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')

% [Vs v_opt]
% [Qsa_lineal_acumulada(:,end) q_opt]
% sum(policy(:,3:end-2)) % comprobación de la policy; deben estar los unos alternos

if strcmp(initType, 'center') % Return esperada si empezamos en el centro
    r = [0 0 0 0 0 1];
    g = 0.9.^[0:5];
    g*r';
elseif strcmp(initType, 'leftCorner') % Return esperada si empezamos en el estado más a la izquierda
    r = [0 0 0 0 0 0 0 0 0 0 1];
    g = 0.9.^[0:10];
    g*r';
end
fprintf('Return esperada: %d\n', g*r')