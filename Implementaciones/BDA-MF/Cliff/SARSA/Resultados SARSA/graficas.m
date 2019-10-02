clc

Gmean_todos = [];
mean_Gmean_eps0_todos = [];

Gmean_todos = [Gmean_todos; Gmean];
mean_Gmean_eps0_todos = [mean_Gmean_eps0_todos; mean_Gmean_eps0];

alfa = [0.10 0.15 0.20 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 1.1];
epsilon = [0.001 0.005 0.01 0.05 0.10 0.15 0.20];

save('Pruebas alfa\Gmean_alpha.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'alfa')
save('Pruebas epsilon\Gmean_epsilon.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'epsilon')
%%
clear all, clc
load('Pruebas alfa\Gmean_alpha.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\alpha = 0.10','\alpha = 0.15','\alpha = 0.20','\alpha = 0.25','\alpha = 0.30'...
    ,'\alpha = 0.35','\alpha = 0.40','\alpha = 0.45','\alpha = 0.50','\alpha = 0.55'...
    ,'\alpha = 0.60','\alpha = 0.65','\alpha = 0.70','\alpha = 0.75','\alpha = 0.80'...
    ,'\alpha = 0.85','\alpha = 0.90','\alpha = 0.95','\alpha = 1','\alpha = 1.1')
title('\alpha test: SARSA return (mean)')

load('Pruebas alfa\Gmean_alpha.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\alpha = 0.10','\alpha = 0.15','\alpha = 0.20','\alpha = 0.25','\alpha = 0.30'...
    ,'\alpha = 0.35','\alpha = 0.40','\alpha = 0.45','\alpha = 0.50','\alpha = 0.55'...
    ,'\alpha = 0.60','\alpha = 0.65','\alpha = 0.70','\alpha = 0.75','\alpha = 0.80'...
    ,'\alpha = 0.85','\alpha = 0.90','\alpha = 0.95','\alpha = 1','\alpha = 1.1')
title('\alpha test: SARSA return \epsilon = 0 (mean)')

%%
clear all, clc
load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\epsilon = 0.001','\epsilon = 0.005','\epsilon = 0.01','\epsilon = 0.05','\epsilon = 0.1','\epsilon = 0.15','\epsilon = 0.2')
title('\epsilon test: SARSA return (mean)')

load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\epsilon = 0.001','\epsilon = 0.005','\epsilon = 0.01','\epsilon = 0.05','\epsilon = 0.1','\epsilon = 0.15','\epsilon = 0.2')
title('\epsilon test: SARSA return \epsilon = 0 (mean)')