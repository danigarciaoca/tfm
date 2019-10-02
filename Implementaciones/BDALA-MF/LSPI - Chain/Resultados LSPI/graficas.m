clear all, close all, clc

Gmean_todos = [];
mean_Gmean_eps0_todos = [];

Gmean_todos = [Gmean_todos; Gmean];
mean_Gmean_eps0_todos = [mean_Gmean_eps0_todos; mean_Gmean_eps0];

max_steps = [50 100 150 200 300];
num_episodes = [50 100 150 200 300];
epsilon = [0.05 0.1 0.15 0.2];

save('Pruebas max_steps\Gmean_max_steps.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'max_steps')
save('Pruebas epsilon\Gmean_epsilon.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'epsilon')
%% max_steps
clear all, clc
load('Pruebas max_steps\Gmean_max_steps.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('max steps = 50','max steps = 100','max steps = 150','max steps = 200','max steps = 300')
title('max steps test: LSPI return (mean)')

load('Pruebas max_steps\Gmean_max_steps.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('max steps = 50','max steps = 100','max steps = 150','max steps = 200','max steps = 300')
title('max steps test: LSPI return \epsilon = 0 (mean)')

%% epsilon
clear all, clc
load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\epsilon = 0.05', '\epsilon = 0.1', '\epsilon = 0.15', '\epsilon = 0.2')
title('\epsilon test: LSPI return (mean)')

load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\epsilon = 0.05', '\epsilon = 0.1', '\epsilon = 0.15', '\epsilon = 0.2')
title('\epsilon test: LSPI return \epsilon = 0 (mean)')

%% num_episodes
clear all, clc
pathRes = 'Pruebas num_episodes\';
numEpisodesAux = [50 100 150 200 300];
figure, hold on
for i = 1:size(numEpisodesAux,2)
    load([pathRes 'nExp=50,nEpi=' num2str(numEpisodesAux(i)) ',steps=200,eps=0.1.mat'])
    plot(1:size(Gmean,2), Gmean, 'LineWidth', 2)
end
hold off, xlim([0 300])
legend('numEpi=50','numEpi=100','numEpi=150','numEpi=200','numEpi=300')
title('numEpi test: LSPI return (mean)')

figure, hold on
for i = 1:size(numEpisodesAux,2)
    load([pathRes 'nExp=50,nEpi=' num2str(numEpisodesAux(i)) ',steps=200,eps=0.1.mat'])
    plot(1:size(Gmean,2), mean_Gmean_eps0*ones(size(Gmean_eps0)), 'LineWidth', 2)
end
hold off, xlim([0 300])
legend('numEpi=50','numEpi=100','numEpi=150','numEpi=200','numEpi=300')
title('numEpi test: LSPI return \epsilon = 0 (mean)')