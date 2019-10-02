clear all, close all, clc

%%
load('Calibración dual-ascent\Resultados exacto\RL_dual_ascent_V_exacta_optimo_2500.mat')
Gmean = mean(G,1); % Media de los experimentos realizados
order = 3; long = 55;
Gsmooth = sgolayfilt(Gmean,order,long);
figure, hold on
plot((1:numRep*numEpisodes), Gmean, 'LineWidth', 2)
plot((1:numRep*numEpisodes), Gsmooth, 'LineWidth', 2)
hold off
title(['DA(EXACT) - Reward per episode (always starting at s=2)'])
xlabel('Episode'), ylabel('G')
ylim([0 40])

errorD_mean = mean(errorD,1); % Media de los experimentos realizados
figure, plot((1:numRep*numEpisodes), errorD_mean, 'LineWidth', 2)
title('DA(EXACT) - Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')
ylim([0 4])

%%
load('2. Gráficas calibración dual-ascent\RL_dual_ascent_optimo.mat')
Gmean = mean(G,1); % Media de los experimentos realizados
order = 3; long = 55;
Gsmooth = sgolayfilt(Gmean,order,long);
figure, hold on
plot((1:numRep*numEpisodes), Gmean, 'LineWidth', 2)
plot((1:numRep*numEpisodes), Gsmooth, 'LineWidth', 2)
hold off
title(['DA(TD) - Reward per episode (always starting at s=2)'])
xlabel('Episode'), ylabel('G')
ylim([0 40])

errorD_mean = mean(errorD,1); % Media de los experimentos realizados
figure, plot((1:numRep*numEpisodes), errorD_mean, 'LineWidth', 2)
title('DA(TD) - Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')
ylim([0 4])

%%
load('3. Gráficas calibración SARSA\SARSA_optimo.mat')
Gmean = mean(G,1); % Media de los experimentos realizados
order = 3; long = 55;
Gsmooth = sgolayfilt(Gmean,order,long);
figure, hold on
plot((1:numEpisodes), Gmean, 'LineWidth', 2)
plot((1:numEpisodes), Gsmooth, 'LineWidth', 2)
hold off
title(['SARSA - Reward per episode (always starting at s=2)'])
xlabel('Episode'), ylabel('G')
ylim([0 40])

figure, plot((1:numEpisodes), policy_error_mean, 'LineWidth', 2)
title('SARSA - Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')
ylim([0 4])

%%
load('4. Gráficas calibración Q-learning\Q-learning_optimo.mat')
Gmean = mean(G,1); % Media de los experimentos realizados
order = 3; long = 55;
Gsmooth = sgolayfilt(Gmean,order,long);
figure, hold on
plot((1:numEpisodes), Gmean, 'LineWidth', 2)
plot((1:numEpisodes), Gsmooth, 'LineWidth', 2)
hold off
title(['QL - Reward per episode (always starting at s=2)'])
xlabel('Episode'), ylabel('G')
ylim([0 40])

figure, plot((1:numEpisodes), policy_error_mean, 'LineWidth', 2)
title('QL - Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')
ylim([0 4])