% errorD_mean_todos = []
% errorD_mean_todos = [errorD_mean_todos;errorD_mean];
% 
% numEpisodes = [20 25 30 35 40 45 50 55 60];
% alphaD = [1 0.8 0.6 0.5 0.4 0.2 0.1 0.08 0.05];
% alphaTD = [1 0.8 0.6 0.5 0.4 0.2 0.1 0.08 0.05];
% epsilon = [0 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.7 0.9 1];
% 
% save('Calibración dual-ascent exacto\Resultados TD_corregido converge (rapido)\errorD_alphaD.mat', 'errorD_mean_todos', 'alphaD')
% save('Calibración dual-ascent exacto\Resultados TD_corregido converge (rapido)\errorD_alphaTD.mat', 'errorD_mean_todos', 'alphaTD')
% save('Calibración dual-ascent exacto\Resultados TD_corregido converge (rapido)\errorD_alphaEpsilon.mat', 'errorD_mean_todos', 'epsilon')

tam_fuent_title = 14;
tam_fuente_label = 14;
tam_fuente_legend = 12;
%%
load('2. Gráficas calibración dual-ascent\Pruebas alfaTD\errorD_alphaTD.mat')
figure, plot([1:size(errorD_mean_todos,2)], errorD_mean_todos, 'LineWidth',2)
h_legend = legend('\alpha_{TD} = 1','\alpha_{TD} = 0.8','\alpha_{TD} = 0.6','\alpha_{TD} = 0.5','\alpha_{TD} = 0.4','\alpha_{TD} = 0.2'...
    ,'\alpha_{TD} = 0.1','\alpha_{TD} = 0.08','\alpha_{TD} = 0.05');
set(h_legend,'FontSize',tam_fuente_legend);
title('Error en la política. Prueba: variación \alpha_{TD}','FontSize',tam_fuent_title)
xlabel('Episodio','FontSize',tam_fuente_label),ylabel('${\left\Vert \pi_{d_{k}}-\pi^{*}\right\Vert _{2}^{2}}$','FontSize',tam_fuente_label,'Interpreter','LaTex')

%%
load('2. Gráficas calibración dual-ascent\Pruebas alfaD\errorD_alphaD.mat')
figure, plot([1:size(errorD_mean_todos,2)], errorD_mean_todos, 'LineWidth',2)
h_legend = legend('\alpha_D = 1','\alpha_D = 0.8','\alpha_D = 0.6','\alpha_D = 0.5','\alpha_D = 0.4','\alpha_D = 0.2'...
    ,'\alpha_D = 0.1','\alpha_D = 0.08','\alpha_D = 0.05');
set(h_legend,'FontSize',tam_fuente_legend);
title('Error en la política. Prueba: variación \alpha_D','FontSize',tam_fuent_title)
xlabel('Episodio','FontSize',tam_fuente_label),ylabel('${\left\Vert \pi_{d_{k}}-\pi^{*}\right\Vert _{2}^{2}}$','FontSize',tam_fuente_label,'Interpreter','LaTex')

%%
load('2. Gráficas calibración dual-ascent\Pruebas epsilon\errorD_epsilon.mat')
figure, plot([1:size(errorD_mean_todos([1 3 5 7 9 11 13 14 16],:),2)], errorD_mean_todos([1 3 5 7 9 11 13 14 16],:), 'LineWidth',2)
% h_legend = legend('\epsilon = 0','\epsilon = 0.03','\epsilon = 0.05','\epsilon = 0.07','\epsilon = 0.1','\epsilon = 0.13'...
%     ,'\epsilon = 0.15','\epsilon = 0.17', '\epsilon = 0.2', '\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.4', '\epsilon = 0.5', '\epsilon = 0.7', '\epsilon = 0.9', '\epsilon = 1');
h_legend = legend('\epsilon = 0','\epsilon = 0.05','\epsilon = 0.1',...
    '\epsilon = 0.15', '\epsilon = 0.2', '\epsilon = 0.3', '\epsilon = 0.5', '\epsilon = 0.7', '\epsilon = 1');
set(h_legend,'FontSize',tam_fuente_legend);
title('Error en la política. Prueba: variación \epsilon','FontSize',tam_fuent_title)
xlabel('Episodio','FontSize',tam_fuente_label),ylabel('${\left\Vert \pi_{d_{k}}-\pi^{*}\right\Vert _{2}^{2}}$','FontSize',tam_fuente_label,'Interpreter','LaTex')

%%
pathRes = '2. Gráficas calibración dual-ascent\Pruebas numEpi\';
numEpisodesAux = [20 25 30 35 40 45 50 55 60];
figure, hold on
for i = 1:size(numEpisodesAux,2)
    load([pathRes 'numExp=50,numRep=50,numEpi=' num2str(numEpisodesAux(i)) ',alphaD=0.2,alphaTD=0.2,epsilon=0.1.mat'])
    plot(1:size(errorD_mean,2), errorD_mean, 'LineWidth', 2)
end
hold off, xlim([0 1500])
h_legend = legend('N_{epi}=20','N_{epi}=25','N_{epi}=30','N_{epi}=35','N_{epi}=40','N_{epi}=45','N_{epi}=50','N_{epi}=55','N_{epi}=60');
set(h_legend,'FontSize',tam_fuente_legend);
title('Error en la política. Prueba: variación N_{epi}','FontSize',tam_fuent_title)
xlabel('Episodio','FontSize',tam_fuente_label),ylabel('${\left\Vert \pi_{d_{k}}-\pi^{*}\right\Vert _{2}^{2}}$','FontSize',tam_fuente_label,'Interpreter','LaTex')

