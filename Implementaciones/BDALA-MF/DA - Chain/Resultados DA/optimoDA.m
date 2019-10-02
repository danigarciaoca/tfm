clear all, clc

% load('DA_optimo_chain_eps=1.mat')
load('DA_optimo_chain_eps=0.2.mat')
% Tanto la de epsilon = 1 como epsilon = 0.2 podrían ser válidas. epsilon =
% 0, no (mucho error de policy)
% load('DA_optimo_chain_eps=0.mat')

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
figure, hold on
plot((1:numRep*numEpisodes), Gmean, 'b', 'LineWidth', 2)
% plot((1:numRep*numEpisodes), Gmean_eps0, 'g', 'LineWidth', 2)
plot((1:numRep*numEpisodes), mean(Gmean_eps0)*ones(size(Gmean_eps0)),'--r', 'LineWidth', 2)
hold off, title(['Return per episode (always starting at s=' ,num2str(env.initial_state), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon)], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0']},'Location','southeast','FontSize',12)
ylim([0 10])


errorD_mean = mean(errorD,1); % Media de los experimentos realizados
figure, plot((1:numRep*numEpisodes), errorD_mean, 'LineWidth', 2)
title('Policy error')
xlabel('Episode'), ylabel('d-d_{opt}')
ylim([0 0.05])