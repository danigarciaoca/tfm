clear all, clc

load('Resultados exacto\RL_dual_ascent_V_exacta_optimo.mat')

Vs_mean = mean(squeeze(Vs_acumulada(:,end,:)),2);
[Qsa_mean, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, game);

% % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
order = 3; long = 55;
Gsmooth = sgolayfilt(Gmean,order,long);
figure, hold on
plot((1:numRep*numEpisodes), Gmean, 'LineWidth', 2)
plot((1:numRep*numEpisodes), Gsmooth, 'LineWidth', 2)
hold off
title(['DA (EXACT) - Return per episode (always starting at s=2)'])
xlabel('Episode'), ylabel('G')
legend('Gmean', 'Gmean smooth (Savitzky–Golay filter)','Location','southeast')
ylim([0 40])

aux1 = sum(Qsa_acumulada);
aux2 = squeeze(aux1)';
sumQ_mean = mean(aux2);
figure, plot((1:numRep*numEpisodes), sumQ_mean, 'LineWidth', 2)
title('DA (EXACT) - \Sigmaq(s,a) for each episode')
xlabel('Episode'), ylabel('\Sigmaq(s,a)')

aux1 = sum(Vs_acumulada);
aux2 = squeeze(aux1)';
sumV_mean = mean(aux2);
figure, plot((1:numRep*numEpisodes), sumV_mean, 'LineWidth', 2)
title('DA (EXACT) - \Sigmav(s) for each episode')
xlabel('Episode'), ylabel('\Sigmav(s)')

aux1 = sum((Qsa_acumulada(game.N_actions+1:end-game.N_actions,:,:)-q_opt(game.N_actions+1:end-game.N_actions,:)).^2 , 1);
aux2 = squeeze(aux1)';
mse_q = sqrt(mean(aux2));
figure, plot((1:numRep*numEpisodes), mse_q, 'LineWidth', 2)
title('DA (EXACT) - Mean-squared error of q(s,a) for each episode')
xlabel('Episode'), ylabel('MSE q(s,a)')

aux1 = sum((Vs_acumulada(2:end-1,:,:)-v_opt(2:end-1,:)).^2 , 1);
aux2 = squeeze(aux1)';
mse_v = sqrt(mean(aux2));
figure, plot((1:numRep*numEpisodes), mse_v, 'LineWidth', 2)
title('DA (EXACT) - Mean-squared error of v(s) for each episode')
xlabel('Episode'), ylabel('MSE v(s)')

errorD_mean = mean(errorD,1); % Media de los experimentos realizados
figure, plot((1:numRep*numEpisodes), errorD_mean, 'LineWidth', 2)
title('DA (EXACT) - Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')
ylim([0 3])