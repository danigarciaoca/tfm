% clear all
% close all
clc

tam_fuent_title = 14;
tam_fuente_label = 14;
tam_fuente_legend = 12;

load('1. Gráficas calibración TD\alf_epi_mse_G_opt.mat')
alf_epi_mse_G_opt = alf_epi_mse_G;
load('1. Gráficas calibración TD\alf_epi_mse_G_optGreedy0.2.mat')
alf_epi_mse_G_optGreedy02 = alf_epi_mse_G;
load('1. Gráficas calibración TD\alf_epi_mse_G_rand.mat')
alf_epi_mse_G_rand = alf_epi_mse_G;

figure, hold on
plot(alf_epi_mse_G_opt(:,1),alf_epi_mse_G_opt(:,2),'LineWidth',2)
plot(alf_epi_mse_G_optGreedy02(:,1),alf_epi_mse_G_optGreedy02(:,2),'LineWidth',2)
plot(alf_epi_mse_G_rand(:,1),alf_epi_mse_G_rand(:,2),'LineWidth',2)
hold off
h_legend=legend('$\pi^*$','$\epsilon-\pi^* (\epsilon=0.2)$', '$\pi$ aleatoria');
set(h_legend,'FontSize',tam_fuente_legend);
set(h_legend,'Interpreter','latex')
xlabel('$\alpha_{_{TD}}$','FontSize',tam_fuente_label,'Interpreter','LaTex'), ylabel('Episodios','FontSize',tam_fuente_label)
title('Episodios hasta convergencia','FontSize',tam_fuent_title)

figure, hold on
plot(alf_epi_mse_G_opt(:,1),alf_epi_mse_G_opt(:,3),'LineWidth',2)
plot(alf_epi_mse_G_optGreedy02(:,1),alf_epi_mse_G_optGreedy02(:,3),'LineWidth',2)
plot(alf_epi_mse_G_rand(:,1),alf_epi_mse_G_rand(:,3),'LineWidth',2)
hold off
h_legend=legend('$\pi^*$','$\epsilon-\pi^* (\epsilon=0.2)$', '$\pi$ aleatoria');
set(h_legend,'FontSize',tam_fuente_legend);
set(h_legend,'Interpreter','latex')
xlabel('$\alpha_{_{TD}}$','FontSize',tam_fuente_label,'Interpreter','LaTex'), ylabel('$\Delta$','FontSize',tam_fuente_label,'Interpreter','LaTex')
title('\Delta cuando se ha alcanzado la convergencia','FontSize',tam_fuent_title)

figure, hold on
plot(alf_epi_mse_G_opt(:,1),alf_epi_mse_G_opt(:,4),'LineWidth',2)
plot(alf_epi_mse_G_optGreedy02(:,1),alf_epi_mse_G_optGreedy02(:,4),'LineWidth',2)
plot(alf_epi_mse_G_rand(:,1),alf_epi_mse_G_rand(:,4),'LineWidth',2)
hold off
h_legend = legend('optimal','optimal-greedy 0.2', 'rand','Location','southeast')
set(h_legend,'FontSize',tam_fuente_legend);
xlabel('\alpha_{TD}','FontSize',tam_fuente_label), ylabel('G (return)','FontSize',tam_fuente_label)
title('Total return when convergence reached','FontSize',tam_fuent_title)