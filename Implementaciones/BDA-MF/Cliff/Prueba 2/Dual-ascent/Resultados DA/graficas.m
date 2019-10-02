clear all, close all, clc

Gmean_todos = [];
mean_Gmean_eps0_todos = [];

Gmean_todos = [Gmean_todos; Gmean];
mean_Gmean_eps0_todos = [mean_Gmean_eps0_todos; mean_Gmean_eps0];

alfaD = [1 0.8 0.5 0.3 0.2 0.1 0.08 0.05 0.01 0.005 0.001 0.0005 0.0001];
alfaTD = [1 0.8 0.5 0.3 0.2 0.1 0.08 0.05 0.01 0.005 0.001 0.0005 0.0001];
epsilon = [0.001 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.9 1];

save('Pruebas alfaD\Gmean_alphaD.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'alfaD')
save('Pruebas alfaTD\Gmean_alphaTD.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'alfaTD')
save('Pruebas epsilon\Gmean_epsilon.mat', 'Gmean_todos', 'mean_Gmean_eps0_todos', 'epsilon')
%%
clear all, clc
load('Pruebas alfaD\Gmean_alphaD.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)

legend('\alpha_D = 1','\alpha_D = 0.8','\alpha_D = 0.5','\alpha_D = 0.3','\alpha_D = 0.2'...
    ,'\alpha_D = 0.1','\alpha_D = 0.08','\alpha_D = 0.05','\alpha_D = 0.01','\alpha_D = 0.005'...
    ,'\alpha_D = 0.001','\alpha_D = 0.0005','\alpha_D = 0.0001')
title('\alpha_D test: DA return (mean)')

load('Pruebas alfaD\Gmean_alphaD.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\alpha_D = 1','\alpha_D = 0.8','\alpha_D = 0.5','\alpha_D = 0.3','\alpha_D = 0.2'...
    ,'\alpha_D = 0.1','\alpha_D = 0.08','\alpha_D = 0.05','\alpha_D = 0.01','\alpha_D = 0.005'...
    ,'\alpha_D = 0.001','\alpha_D = 0.0005','\alpha_D = 0.0001')
title('\alpha_D test: DA return \epsilon = 0 (mean)')
%%
clear all, clc
load('Pruebas alfaTD\Gmean_alphaTD.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\alpha_{TD} = 1','\alpha_{TD} = 0.8','\alpha_{TD} = 0.5','\alpha_{TD} = 0.3','\alpha_{TD} = 0.2'...
    ,'\alpha_{TD} = 0.1','\alpha_{TD} = 0.08','\alpha_{TD} = 0.05','\alpha_{TD} = 0.01','\alpha_{TD} = 0.005'...
    ,'\alpha_{TD} = 0.001','\alpha_{TD} = 0.0005','\alpha_{TD} = 0.0001')
title('\alpha_{TD} test: DA return (mean)')

load('Pruebas alfaTD\Gmean_alphaTD.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\alpha_{TD} = 1','\alpha_{TD} = 0.8','\alpha_{TD} = 0.5','\alpha_{TD} = 0.3','\alpha_{TD} = 0.2'...
    ,'\alpha_{TD} = 0.1','\alpha_{TD} = 0.08','\alpha_{TD} = 0.05','\alpha_{TD} = 0.01','\alpha_{TD} = 0.005'...
    ,'\alpha_{TD} = 0.001','\alpha_{TD} = 0.0005','\alpha_{TD} = 0.0001')
title('\alpha_{TD} test: DA return \epsilon = 0 (mean)')
%%
clear all, clc

load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], Gmean_todos, 'LineWidth',2)
legend('\epsilon = 0.001', '\epsilon = 0.01', '\epsilon = 0.05', '\epsilon = 0.1', '\epsilon = 0.15', '\epsilon = 0.2'...
    ,'\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.35', '\epsilon = 0.4', '\epsilon = 0.45'...
    ,'\epsilon = 0.5', '\epsilon = 0.6', '\epsilon = 0.7', '\epsilon = 0.9', '\epsilon = 1')
title('\epsilon test: DA return (mean)')

load('Pruebas epsilon\Gmean_epsilon.mat')
figure, plot([1:size(Gmean_todos,2)], mean_Gmean_eps0_todos*ones(1,size(Gmean_todos,2)), 'LineWidth',2)
legend('\epsilon = 0.001', '\epsilon = 0.01', '\epsilon = 0.05', '\epsilon = 0.1', '\epsilon = 0.15', '\epsilon = 0.2'...
    ,'\epsilon = 0.25', '\epsilon = 0.3', '\epsilon = 0.35', '\epsilon = 0.4', '\epsilon = 0.45'...
    ,'\epsilon = 0.5', '\epsilon = 0.6', '\epsilon = 0.7', '\epsilon = 0.9', '\epsilon = 1')
title('\epsilon test: DA return \epsilon = 0 (mean)')

%%
clear all, clc
pathRes = 'Pruebas nEpi\';
numEpisodesAux = [1 10 20 25 30 35 40 45 50 55 60];
figure, hold on
for i = 1:size(numEpisodesAux,2)
    load([pathRes 'nExp=20,nRep=100,nEpi=' num2str(numEpisodesAux(i)) ',alphaD=0.01,alphaTD=0.01,eps=0.3.mat'])
    plot(1:size(Gmean,2), Gmean, 'LineWidth', 2)
end
hold off, xlim([0 1000])
legend('numEpi=1','numEpi=10','numEpi=20','numEpi=25','numEpi=30','numEpi=35','numEpi=40','numEpi=45','numEpi=50','numEpi=55','numEpi=60')
title('numEpi test: DA return (mean)')

figure, hold on
for i = 1:size(numEpisodesAux,2)
    load([pathRes 'nExp=20,nRep=100,nEpi=' num2str(numEpisodesAux(i)) ',alphaD=0.01,alphaTD=0.01,eps=0.3.mat'])
    plot(1:size(Gmean,2), mean_Gmean_eps0*ones(size(Gmean_eps0)), 'LineWidth', 2)
end
hold off, xlim([0 1000])
legend('numEpi=1','numEpi=10','numEpi=20','numEpi=25','numEpi=30','numEpi=35','numEpi=40','numEpi=45','numEpi=50','numEpi=55','numEpi=60')
title('numEpi test: DA return \epsilon = 0 (mean)')

%%
clear all, clc
pathRes = 'Pruebas nRep\';
numRepAux = [10 20 30 40 50 60 70 80 90 100 150 200 250 300];;
figure, hold on
for i = 1:size(numRepAux,2)
    load([pathRes 'nExp=20,nRep=' num2str(numRepAux(i)) ',nEpi=20,alphaD=0.1,alphaTD=0.08,eps=0.1.mat'])
    plot(1:size(Gmean,2), Gmean, 'LineWidth', 2)
end
hold off, xlim([0 1000])
legend('numRep=10','numRep=20','numRep=30','numRep=40','numRep=50','numRep=60'...
    ,'numRep=70','numRep=80','numRep=90','numRep=100','numRep=150','numRep=200','numRep=250','numRep=300')
title('numRep test: DA return (mean)')

figure, hold on
for i = 1:size(numRepAux,2)
    load([pathRes 'nExp=20,nRep=' num2str(numRepAux(i)) ',nEpi=20,alphaD=0.1,alphaTD=0.08,eps=0.1.mat'])
    plot(1:size(Gmean,2), mean_Gmean_eps0*ones(size(Gmean_eps0)), 'LineWidth', 2)
end
hold off, xlim([0 1000])
legend('numRep=10','numRep=20','numRep=30','numRep=40','numRep=50','numRep=60'...
    ,'numRep=70','numRep=80','numRep=90','numRep=100','numRep=150','numRep=200','numRep=250','numRep=300')
title('numRep test: DA return \epsilon = 0 (mean)')