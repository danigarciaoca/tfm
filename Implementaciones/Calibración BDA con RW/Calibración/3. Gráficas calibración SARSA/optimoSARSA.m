clear all, clc

load('SARSA_optimo.mat')

% getPolicy obtiene la policy del promedio de las Q(s,a) de todos los
% experimentos realizados
policy = getPolicy(Qsa_lineal_acumulada, game)
Qsa % última Q(s,a) calculada
Vs = getValueFunction(Qsa, policy, game) %de la última Q(s,a) calculada

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
figure, plot((1:numEpisodes), Gmean, 'LineWidth', 2)
title(['SARSA - Reward per episode (always starting at s=2)'])
xlabel('Episode'), ylabel('G')
ylim([0 30])

aux1 = sum(Qsa_lineal_acumulada);
aux2 = squeeze(aux1)';
sumQ_mean = mean(aux2);
figure, plot((1:numEpisodes), sumQ_mean, 'LineWidth', 2)
title('SARSA - \Sigmaq(s,a) for each episode')
xlabel('Episode'), ylabel('\Sigmaq(s,a)')

aux1 = sum(Vs_acumulada);
aux2 = squeeze(aux1)';
sumV_mean = mean(aux2);
figure, plot((1:numEpisodes), sumV_mean, 'LineWidth', 2)
title('SARSA - \Sigmav(s) for each episode')
xlabel('Episode'), ylabel('\Sigmav(s)')

aux1 = sum((Qsa_lineal_acumulada(game.N_actions+1:end-game.N_actions,:,:)-q_opt(game.N_actions+1:end-game.N_actions,:)).^2 , 1);
aux2 = squeeze(aux1)';
mse_q = sqrt(mean(aux2));
figure, plot((1:numEpisodes), mse_q, 'LineWidth', 2)
title('SARSA - Mean-squared error of q(s,a) for each episode')
xlabel('Episode'), ylabel('MSE q(s,a)')

aux1 = sum((Vs_acumulada(2:end-1,:,:)-v_opt(2:end-1,:)).^2 , 1);
aux2 = squeeze(aux1)';
mse_v = sqrt(mean(aux2));
figure, plot((1:numEpisodes), mse_v, 'LineWidth', 2)
title('SARSA - Mean-squared error of v(s) for each episode')
xlabel('Episode'), ylabel('MSE v(s)')

% Calcula el error en la policy en cada episodio, promediado entre todos
% los experimentos
policy_vector = getPolicyVector(Qsa_lineal_acumulada, game); % policy de cada experimento y para cada iteración
optimal_policy_vector = getPolicyVector(q_opt, game); % policy óptima
errorPolicy = policy_vector - optimal_policy_vector; % error de policies
dif = policy_vector(game.N_actions+1:end-game.N_actions,:,:) - optimal_policy_vector(game.N_actions+1:end-game.N_actions);
policy_error_mean = mean(sqrt(sum(dif.^2)),3); % ecm y promedio de entre todos los experimentos

figure, plot((1:numEpisodes), policy_error_mean, 'LineWidth', 2)
title('SARSA - Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')
ylim([0 4])

% [Vs v_opt]
% [Qsa_lineal_acumulada(:,end) q_opt]
% sum(policy(:,3:end-2)) % comprobación de la policy; deben estar los unos alternos