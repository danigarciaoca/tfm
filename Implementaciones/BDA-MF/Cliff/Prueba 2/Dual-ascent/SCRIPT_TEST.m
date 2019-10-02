clear all, close all, clc
max_steps = 1000; % Número máximo de pasos en cada episodio
%% Pruebas numRep
% clear all, close all, clc
numExperiments = 20; % Número de experimentos de numRep*numEpisodes episodios
numRep = [10 20 30 40 50 60 70 80 90 100 150 200 250 300];
numEpisodes = 20;
alphaD = 0.01;
alphaTD = 0.01;
epsilon = 0.3;
rltv_path = 'Resultados DA\Pruebas nRep';
for i = 1:size(numRep,2)
    i
    RL_function(numExperiments, numRep(i), numEpisodes, max_steps, epsilon, alphaD, alphaTD, rltv_path)
end

%% Pruebas numEpi
% clear all, close all, clc
numExperiments = 20; % Número de experimentos de numRep*numEpisodes episodios
numRep = 100; % Número de repeticiones de cada set de episodios
numEpisodes = [1 10 20 25 30 35 40 45 50 55 60];
alphaD = 0.01;
alphaTD = 0.01;
epsilon = 0.3;
rltv_path = 'Resultados DA\Pruebas nEpi';
for i = 1:size(numEpisodes,2)
    i
    RL_function(numExperiments, numRep, numEpisodes(i), max_steps, epsilon, alphaD, alphaTD, rltv_path)
end

%% Pruebas alphaD
% clear all, close all, clc
numExperiments = 20; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 50;
alphaD = [1 0.8 0.5 0.3 0.2 0.1 0.08 0.05 0.01 0.005 0.001 0.0005 0.0001];
alphaTD = 0.01;
epsilon = 0.3;
rltv_path = 'Resultados DA\Pruebas alfaD';
for i = 1:size(alphaD,2)
    i
    RL_function(numExperiments, numRep, numEpisodes, max_steps, epsilon, alphaD(i), alphaTD, rltv_path)
end

%% Pruebas alphaTD
% clear all, close all, clc
numExperiments = 20; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 20;
alphaD = 0.01;
alphaTD = [1 0.8 0.5 0.3 0.2 0.1 0.08 0.05 0.01 0.005 0.001 0.0005 0.0001];
epsilon = 0.3;
rltv_path = 'Resultados DA\Pruebas alfaTD';
for i = 4:size(alphaTD,2)
    i
    RL_function(numExperiments, numRep, numEpisodes, max_steps, epsilon, alphaD, alphaTD(i), rltv_path)
end

%% Pruebas epsilon
% clear all, close all, clc
numExperiments = 20; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 20;
alphaD = 0.01;
alphaTD = 0.01;
epsilon = [0.001 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.9 1];
rltv_path = 'Resultados DA\Pruebas epsilon';
nExp=20,nRep=50,nEpi=20,alphaD=0.01,alphaTD=0.01,eps=0.1
for i = 1:size(epsilon,2)
    i
    RL_function(numExperiments, numRep, numEpisodes, max_steps, epsilon(i), alphaD, alphaTD, rltv_path)
end