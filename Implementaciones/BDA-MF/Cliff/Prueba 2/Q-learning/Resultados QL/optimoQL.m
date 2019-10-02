clear all, clc

load('Q-learning_optimo_cliff.mat')

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
figure, hold on
plot((1:numEpisodes), Gmean, 'b', 'LineWidth', 2)
% plot((1:numEpisodes), Gmean_eps0, 'g', 'LineWidth', 2)
plot((1:numEpisodes), mean(Gmean_eps0)*ones(size(Gmean_eps0)),'--r', 'LineWidth', 2)
hold off, title(['QL - Return per episode (always starting at s=' ,num2str(env.initState), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon) ' - QL'], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0 - QL']},'Location','southeast','FontSize',12)

checkCliffResults(env, q_opt_format, Qsa_and_policy_opt)

% Return que cabría esperar en el problema del cliff con la política óptima
g = gamma.^([1:13]-1);
r = -1*ones(1,13);
g*r'