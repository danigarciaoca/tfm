% errorD_mean_todos = [];
% errorD_mean_todos = [errorD_mean_todos;errorD_mean];

% alphaD = [1 0.8 0.6 0.5 0.4 0.2 0.1 0.08 0.05];
% alphaTD = [1 0.8 0.6 0.5 0.4 0.2 0.1 0.08 0.05];
% epsilon = [0 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.7 0.9 1];

% save('Resultados exacto\Pruebas alfaD\errorD_alphaD.mat', 'errorD_mean_todos', 'alphaD')
% save('Resultados exacto\Pruebas alfaTD\errorD_alphaTD.mat', 'errorD_mean_todos', 'alphaTD')
% save('Resultados exacto\Pruebas epsilon\errorD_alphaEpsilon.mat', 'errorD_mean_todos', 'epsilon')


%%
load('Resultados exacto\Pruebas alfaTD\errorD_alphaTD.mat')
figure, plot([1:size(errorD_mean_todos,2)], errorD_mean_todos, 'LineWidth',2)
legend('\alpha_{TD} = 1','\alpha_{TD} = 0.8','\alpha_{TD} = 0.6','\alpha_{TD} = 0.5','\alpha_{TD} = 0.4','\alpha_{TD} = 0.2'...
    ,'\alpha_{TD} = 0.1','\alpha_{TD} = 0.08','\alpha_{TD} = 0.05')
title('Policy error. Pruebas \alpha_{TD} (RL-EXACT)')
ylim([0 2.5])

%%
load('Resultados exacto\Pruebas alfaD\errorD_alphaD.mat')
figure, plot([1:size(errorD_mean_todos,2)], errorD_mean_todos, 'LineWidth',2)
legend('\alpha_D = 1','\alpha_D = 0.8','\alpha_D = 0.6','\alpha_D = 0.5','\alpha_D = 0.4','\alpha_D = 0.2'...
    ,'\alpha_D = 0.1','\alpha_D = 0.08','\alpha_D = 0.05')
title('Policy error. Pruebas \alpha_D (RL-EXACT)')
ylim([0 2.5])

%%
load('Resultados exacto\Pruebas epsilon\errorD_epsilon.mat')
figure, plot([1:size(errorD_mean_todos,2)], errorD_mean_todos, 'LineWidth',2)
legend('\epsilon = 0','\epsilon = 0.03','\epsilon = 0.05','\epsilon = 0.07','\epsilon = 0.1','\epsilon = 0.13'...
    ,'\epsilon = 0.15','\epsilon = 0.17', '\epsilon = 0.2', '\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.4', '\epsilon = 0.5', '\epsilon = 0.7', '\epsilon = 0.9', '\epsilon = 1')
title('Policy error. Pruebas \epsilon (RL-EXACT)')
ylim([0 2.5])

%%
pathRes = 'Resultados exacto\Pruebas numEpi\';
numEpisodesAux = [1 10 20 25 30 35 40 45 50 55 60];
figure, hold on
for i = 1:size(numEpisodesAux,2)
    load([pathRes 'numExperiments=50,numRep=50,numEpisodes=' num2str(numEpisodesAux(i)) ',alphaD=0.2,alphaTD=0.2,epsilon=0.1.mat'])
    plot(1:size(errorD_mean,2), errorD_mean, 'LineWidth', 2)
end
hold off, xlim([0 1500])
legend('numEpi=1','numEpi=10','numEpi=20','numEpi=25','numEpi=30','numEpi=35','numEpi=40','numEpi=45','numEpi=50','numEpi=55','numEpi=60')
title('Policy error. Pruebas numEpi (RL-EXACT)')
ylim([0 2.5])