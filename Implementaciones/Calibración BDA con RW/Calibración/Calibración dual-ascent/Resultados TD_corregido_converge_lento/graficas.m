% errorD_mean_todos = []
% errorD_mean_todos = [errorD_mean_todos;errorD_mean];
% % alphaD = [1 0.1 0.01 0.001 0.0001 0.00001];
% % alphaD = [1 0.1 0.05 0.01 0.005 0.001 0.0001 0.00001];
% % alphaD = [1 0.7 0.5 0.3 0.1 0.07 0.05 0.03 0.01 0.005 0.001];
% % alphaTD = [1 0.7 0.5 0.3 0.1 0.07 0.05];
% % epsilon = [0 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.9];
% % alphaTD = [1 0.7 0.5 0.3 0.1 0.07 0.05 0.03 0.01];
% alphaD = [0.0005 0.0001 0.00005 0.00001];
% save('errorD_alphaD_100.mat', 'errorD_mean_todos', 'alphaD')

load('Resultados TD_corregido_converge_lento\Pruebas epsilon\errorD_epsilon.mat')
figure, plot([1:size(errorD_mean_todos,2)], errorD_mean_todos, 'LineWidth',2)
legend('\epsilon = 0','\epsilon = 0.03','\epsilon = 0.05','\epsilon = 0.07','\epsilon = 0.1','\epsilon = 0.13'...
    ,'\epsilon = 0.15','\epsilon = 0.17', '\epsilon = 0.2', '\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.4', '\epsilon = 0.5', '\epsilon = 0.9')

load('Resultados TD_corregido_converge_lento\Pruebas alphaTD\errorD_alphaTD.mat')
figure, plot([1:size(errorD_mean_todos,2)], errorD_mean_todos, 'LineWidth',2)
legend('\alpha_TD = 1','\alpha_TD = 0.7','\alpha_TD = 0.5','\alpha_TD = 0.3','\alpha_TD = 0.1','\alpha_TD = 0.07'...
    ,'\alpha_TD = 0.05','\alpha_TD = 0.03','\alpha_TD = 0.01')

load('Resultados TD_corregido_converge_lento\Pruebas alphaD (100 episodios)\errorD_alphaD_100.mat')
figure, plot([1:size(errorD_mean_todos,2)], errorD_mean_todos, 'LineWidth',2)
legend('\alpha_D = 0.0005','\alpha_D = 0.0001','\alpha_D = 0.00005','\alpha_D = 0.00001')
