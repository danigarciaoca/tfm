clear all, clc

% % % % ENTORNO:
N_states = 13;
transitionType = 'det'; 
rewardType = 'det';
game = random_walk_set_up_mod(N_states, transitionType, rewardType);

% Optimum value
v_opt=inv(eye(game.N_states)-game.gamma*game.pi_opt*game.P)*game.pi_opt*game.R;
q_opt=inv(eye(game.N_states*game.N_actions)-game.gamma*game.P*game.pi_opt)*game.R;

% % % % AGENTE:
numExperiments = 30; % N�mero de experimentos sobre el que se promedia
numEpisodes = 1000; % N�mero de episodios de cada experimento
maxNumStepsPerEpisode = 50; % N�mero m�ximo de pasos en cada episodio
G = zeros(numExperiments,numEpisodes); % Reward por episodio
G_eps0_acumulada = zeros(numExperiments,numEpisodes); % Return por episodio cuando epsilon = 0
epsilon = 0; % e-greedy value
alpha = 0.7;

% Variable que acumular� las funciones Q y V al final de cada episodio
Qsa_lineal_acumulada = zeros(game.N_states*game.N_actions, numEpisodes, numExperiments);
Vs_acumulada = zeros(game.N_states, numEpisodes, numExperiments);

for exp = 1:numExperiments
    % Inicializamos Q para cada experimento
    % Qsa = zeros(game.N_states, game.N_actions);
    % Inicializamos Q para cada experimento
    Qsa = rand(game.N_states, game.N_actions);
    Qsa(1,:) = zeros(size(Qsa(1,:)));
    Qsa(end,:) = zeros(size(Qsa(1,:)));
    exp
    
    init_state = 2;
    for i = 1:numEpisodes
        % Inicializamos S
        % currentState = game.centralState;
        currentState = init_state; % empezar en el de la izquierda
        stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpisodes episodes simulated
        
        % Escogemos A (currentAction) de S (currentState) seg�n la e-greedy policy.
        currentAction = e_greedy(Qsa, epsilon, currentState, game.N_actions);
        
        while true % siempre que el episodio no haya terminado
            
            % Tomamos la acci�n A (currentAction), observamos la recompensa R
            % (reward) y el siguiente estado S' (nextState).
            [nextState, reward] = getNextState(game, currentState, currentAction);
            
            % G(exp,i) = reward+game.gamma*G(exp,i); % return following the initial state (si el estado inicial fuese aleatorio, habr�a que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
            G(exp,i) = (game.gamma^stepPerEpisode)*reward+G(exp,i);
            
            % Escogemos A' (nextAction) de S' (nextState) seg�n la e-greedy policy.
            nextAction = e_greedy(Qsa, epsilon, nextState, game.N_actions);
            
            % Actualizamos el valor de Q(s,a)
            Qsa(currentState, currentAction) = Qsa(currentState, currentAction) + alpha*(reward + game.gamma*Qsa(nextState, nextAction) - Qsa(currentState, currentAction));
            
            % Actualizamos los valores
            currentState = nextState;
            currentAction = nextAction;
            stepPerEpisode = stepPerEpisode + 1;
            
            % Evaluaci�n de si el episodio ha terminado o no
            if sum(currentState == game.finalState) == 1 || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
                break; % terminamos el episodio
            end
        end
        % Acumulamos el valor de la funci�n Q(s,a
        Qsa_lineal_acumulada(:, i, exp) = reshape(Qsa', [game.N_states*game.N_actions 1]);
        
        % Acumulamos el valor de la funci�n V(s)
        policy = getPolicy(Qsa, game);
        Vs_acumulada(:, i, exp) = getValueFunction(Qsa, policy, game);
    end
    % Once the policy/Q(s,a) has converged, we evaluate the problem with
    % epsilon = 0 (i.e, greedy policy)
    [~, ~, G_eps0] = sarsa_core(game, Qsa, 0, alpha, numEpisodes, maxNumStepsPerEpisode, init_state);
    G_eps0_acumulada(exp,:) = G_eps0;
end

% getPolicy obtiene la policy del promedio de las Qsa de todos los
% experimentos realizados
policy = getPolicy(Qsa_lineal_acumulada, game);
Vs = getValueFunction(Qsa, policy, game);

% % % REPRESENTACI�N DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
figure, hold on
plot((1:numEpisodes), Gmean, 'b', 'LineWidth', 2)
plot((1:numEpisodes), Gmean_eps0, 'g', 'LineWidth', 2)
plot((1:numEpisodes), mean(Gmean_eps0)*ones(size(Gmean_eps0)),'--r', 'LineWidth', 2)
hold off, title(['Return per episode (always starting at s=' ,num2str(init_state), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon)], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0']},'Location','southeast','FontSize',12)

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
% aux1 = sum((Qsa_lineal_acumulada(game.N_actions+1:end-game.N_actions,:,:)-q_opt(game.N_actions+1:end-game.N_actions,:)).^2 , 1);
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
policy_vector = getPolicyVector(Qsa_lineal_acumulada, game); % policy de cada experimento y para cada iteraci�n
optimal_policy_vector = getPolicyVector(q_opt, game); % policy �ptima
errorPolicy = policy_vector - optimal_policy_vector; % error de policies
dif = policy_vector(game.N_actions+1:end-game.N_actions,:,:) - optimal_policy_vector(game.N_actions+1:end-game.N_actions);
policy_error_mean = mean(sqrt(sum(dif.^2)),3); % ecm y promedio de entre todos los experimentos

figure, plot((1:numEpisodes), policy_error_mean, 'LineWidth', 2)
title('Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')

% [Vs v_opt]
% [Qsa_lineal_acumulada(:,end) q_opt]
% sum(policy(:,3:end-2)) % comprobaci�n de la policy; deben estar los unos alternos
