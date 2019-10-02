clc

% errorPolicy_mean_todos = [];
% Gmean_todos = [];
% 
% errorPolicy_mean_todos = [errorPolicy_mean_todos; policy_error_mean];
% Gmean_todos = [Gmean_todos; Gmean];
% 
% alpha = [1 0.7 0.5 0.3 0.1 0.07 0.05 0.03 0.01 0.005];
% epsilon = [0 0.01 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.9];
% 
% save('errorPolicy_epsilon.mat', 'errorPolicy_mean_todos', 'epsilon')
% save('Gmean_epsilon.mat', 'Gmean_todos', 'epsilon')

%%
load('3. Gráficas calibración SARSA\Pruebas alfa\errorPolicy_alpha.mat')
figure, plot([1:size(errorPolicy_mean_todos,2)], errorPolicy_mean_todos, 'LineWidth',2)
legend('\alpha = 1','\alpha = 0.7','\alpha = 0.5','\alpha = 0.3','\alpha = 0.1','\alpha = 0.07'...
    ,'\alpha = 0.05','\alpha = 0.03', '\alpha = 0.01', '\alpha = 0.005')
title('Policy error. Pruebas \alpha (SARSA)')

% load('Resultados SARSA\Pruebas alfa\Gmean_alpha.mat')
% figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
% legend('\alpha = 1','\alpha = 0.7','\alpha = 0.5','\alpha = 0.3','\alpha = 0.1','\alpha = 0.07'...
%     ,'\alpha = 0.05','\alpha = 0.03', '\alpha = 0.01', '\alpha = 0.005')
% title('Return (mean)')

%%
load('3. Gráficas calibración SARSA\Pruebas epsilon\errorPolicy_epsilon.mat')
figure, plot([1:size(errorPolicy_mean_todos,2)], errorPolicy_mean_todos, 'LineWidth',2)
legend('\epsilon = 0', '\epsilon = 0.01', '\epsilon = 0.03','\epsilon = 0.05','\epsilon = 0.07','\epsilon = 0.1','\epsilon = 0.13'...
    ,'\epsilon = 0.15','\epsilon = 0.17', '\epsilon = 0.2', '\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.4', '\epsilon = 0.5', '\epsilon = 0.9')
title('Policy error. Pruebas \epsilon (SARSA)')
epsilon = [0 0.01 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.9];