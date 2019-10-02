clear all, close all, clc

Gmean_todos = [];
mean_Gmean_eps0_todos = [];

Gmean_todos = [Gmean_todos; Gmean];
mean_Gmean_eps0_todos = [mean_Gmean_eps0_todos; mean_Gmean_eps0];

alfa = [1 0.7 0.5 0.3 0.1 0.07 0.05 0.03 0.01 0.005];
epsilon = [0 0.01 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.9];

save('Pruebas alfa\Gmean_alpha.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'alfa')
save('Pruebas epsilon\Gmean_epsilon.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'epsilon')
%%
load('Pruebas alfa\Gmean_alpha.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\alpha = 1','\alpha = 0.7','\alpha = 0.5','\alpha = 0.3','\alpha = 0.1'...
       ,'\alpha = 0.07','\alpha = 0.05','\alpha = 0.03','\alpha = 0.01','\alpha = 0.005')
title('\alpha test: QL return (mean)')

load('Pruebas alfa\Gmean_alpha.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\alpha = 1','\alpha = 0.7','\alpha = 0.5','\alpha = 0.3','\alpha = 0.1'...
       ,'\alpha = 0.07','\alpha = 0.05','\alpha = 0.03','\alpha = 0.01','\alpha = 0.005')
title('\alpha test: QL return \epsilon = 0 (mean)')
%%
clc
load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\epsilon = 0.0', '\epsilon = 0.01', '\epsilon = 0.03', '\epsilon = 0.05', '\epsilon = 0.07',...
       '\epsilon = 0.1', '\epsilon = 0.13', '\epsilon = 0.15', '\epsilon = 0.17', '\epsilon = 0.2',...
       '\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.4', '\epsilon = 0.5', '\epsilon = 0.9')
title('\epsilon test: QL return (mean)')

load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\epsilon = 0.0', '\epsilon = 0.01', '\epsilon = 0.03', '\epsilon = 0.05', '\epsilon = 0.07',...
       '\epsilon = 0.1', '\epsilon = 0.13', '\epsilon = 0.15', '\epsilon = 0.17', '\epsilon = 0.2',...
       '\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.4', '\epsilon = 0.5', '\epsilon = 0.9')
title('\epsilon test: QL return \epsilon = 0 (mean)')