clear all, close all, clc

Gmean_todos = [];
mean_Gmean_eps0_todos = [];

Gmean_todos = [Gmean_todos; Gmean];
mean_Gmean_eps0_todos = [mean_Gmean_eps0_todos; mean_Gmean_eps0];

alfaD = [1 0.8 0.6 0.5 0.4 0.2 0.1 0.08 0.06 0.05 0.04 0.02];
maxStepsEpisode = [5 10 20 30 40 50 70 100 150 200];
epsilon = [0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 0.9 1];

save('Pruebas alfaD\Gmean_alphaD.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'alfaD')
save('Pruebas step\Gmean_step.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'maxStepsEpisode')
save('Pruebas epsilon\Gmean_epsilon.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'epsilon')
%%
clear all, clc
load('Pruebas alfaD\Gmean_alphaD.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\alpha_D = 1','\alpha_D = 0.8','\alpha_D = 0.6'...
    ,'\alpha_D = 0.5','\alpha_D = 0.4','\alpha_D = 0.2','\alpha_D = 0.1','\alpha_D = 0.08'...
    ,'\alpha_D = 0.06','\alpha_D = 0.05','\alpha_D = 0.04','\alpha_D = 0.02')
title('\alpha_D test: DA return (mean)')

load('Pruebas alfaD\Gmean_alphaD.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\alpha_D = 1','\alpha_D = 0.8','\alpha_D = 0.6'...
    ,'\alpha_D = 0.5','\alpha_D = 0.4','\alpha_D = 0.2','\alpha_D = 0.1','\alpha_D = 0.08'...
    ,'\alpha_D = 0.06','\alpha_D = 0.05','\alpha_D = 0.04','\alpha_D = 0.02')
title('\alpha_D test: DA return \epsilon = 0 (mean)')
%%
clear all, clc
load('Pruebas step\Gmean_step.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('step = 5','step = 10','step = 20','step = 30','step = 40','step = 50'...
    ,'step = 70','step = 100','step = 150','step = 200')
title('step test: DA return (mean)')

load('Pruebas step\Gmean_step.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('step = 5','step = 10','step = 20','step = 30','step = 40','step = 50'...
    ,'step = 70','step = 100','step = 150','step = 200')
title('step test: DA return \epsilon = 0 (mean)')
%%
clear all, clc
load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\epsilon = 0', '\epsilon = 0.05', '\epsilon = 0.1', '\epsilon = 0.15', '\epsilon = 0.2'...
    ,'\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.4'...
    ,'\epsilon = 0.5', '\epsilon = 0.7', '\epsilon = 0.9', '\epsilon = 1')
title('\epsilon test: DA return (mean)')

load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\epsilon = 0', '\epsilon = 0.05', '\epsilon = 0.1', '\epsilon = 0.15', '\epsilon = 0.2'...
    ,'\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.4'...
    ,'\epsilon = 0.5', '\epsilon = 0.7', '\epsilon = 0.9', '\epsilon = 1')
title('\epsilon test: DA return \epsilon = 0 (mean)')

%%
clear all, clc
pathRes = 'Pruebas nEpi\';
numEpisodesAux = [1 10 20 25 30 35 40 45 50 55 60];
figure, hold on
for i = 1:size(numEpisodesAux,2)
    load([pathRes 'nExp=50,nRep=50,nEpi=' num2str(numEpisodesAux(i)) ',alphaD=0.2,eps=0.1,nStepEpi=50.mat'])
    plot(1:size(Gmean,2), Gmean, 'LineWidth', 2)
end
hold off, xlim([0 1000])
legend('numEpi=1','numEpi=10','numEpi=20','numEpi=25','numEpi=30','numEpi=35','numEpi=40','numEpi=45','numEpi=50','numEpi=55','numEpi=60')
title('numEpi test: DA return (mean)')

figure, hold on
for i = 1:size(numEpisodesAux,2)
    load([pathRes 'nExp=50,nRep=50,nEpi=' num2str(numEpisodesAux(i)) ',alphaD=0.2,eps=0.1,nStepEpi=50.mat'])
    plot(1:size(Gmean,2), mean_Gmean_eps0*ones(size(Gmean_eps0)), 'LineWidth', 2)
end
hold off, xlim([0 1000])
legend('numEpi=1','numEpi=10','numEpi=20','numEpi=25','numEpi=30','numEpi=35','numEpi=40','numEpi=45','numEpi=50','numEpi=55','numEpi=60')
title('numEpi test: DA return \epsilon = 0 (mean)')


%%
clear all, clc
pathRes = 'Pruebas nRep\';
numRepAux = [1 10 20 25 30 35 40 45 50 55 60 65 70 80 90 100];
figure, hold on
for i = 1:size(numRepAux,2)
    load([pathRes 'nExp=50,nRep=' num2str(numRepAux(i)) ',nEpi=10,alphaD=0.2,eps=0.1,nStepEpi=50.mat'])
    plot(1:size(Gmean,2), Gmean, 'LineWidth', 2)
end
hold off, xlim([0 1000])
legend('numRep=1','numRep=10','numRep=20','numRep=25','numRep=30','numRep=35','numRep=40','numRep=45','numRep=50','numRep=55','numRep=60'...
    ,'numRep=65','numRep=70','numRep=80','numRep=90','numRep=100')
title('numRep test: DA return (mean)')

figure, hold on
for i = 1:size(numRepAux,2)
    load([pathRes 'nExp=50,nRep=' num2str(numRepAux(i)) ',nEpi=10,alphaD=0.2,eps=0.1,nStepEpi=50.mat'])
    plot(1:size(Gmean,2), mean_Gmean_eps0*ones(size(Gmean_eps0)), 'LineWidth', 2)
end
hold off, xlim([0 1000])
legend('numRep=1','numRep=10','numRep=20','numRep=25','numRep=30','numRep=35','numRep=40','numRep=45','numRep=50','numRep=55','numRep=60'...
    ,'numRep=65','numRep=70','numRep=80','numRep=90','numRep=100')
title('numRep test: DA return \epsilon = 0 (mean)')