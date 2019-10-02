clear all
close all
clc

addpath('env_def','buffer')

%% Pruebas numEpi
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas nEpi';
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 200; % Número de repeticiones de cada set de episodios
numEpisodes = [1 10 20 25 30 35 40];
alphaD = 0.0001;
maxStepsEpisode = 500;
epsilon = 0.01;
for i = 1:size(numEpisodes,2)
    i
    RL_mount_buffer_function(numExperiments, numRep, numEpisodes(i), alphaD, epsilon, maxStepsEpisode, savePath)
end

%% Pruebas alphaD
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas alfaD';
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 200; % Número de repeticiones de cada set de episodios
numEpisodes = 1;
alphaD = [0.2 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001];
maxStepsEpisode = 500;
epsilon = 0.01;
for i = 2:size(alphaD,2)
    RL_mount_buffer_function(numExperiments, numRep, numEpisodes, alphaD(i), epsilon, maxStepsEpisode, savePath)
end

%% Pruebas maxStepsEpisode
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas step';
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 200; % Número de repeticiones de cada set de episodios
numEpisodes = 1;
alphaD = 0.0001;
maxStepsEpisode = [300 400 500 700];
epsilon = 0.01;
for i = 1:size(maxStepsEpisode,2)
    RL_mount_buffer_function(numExperiments, numRep, numEpisodes, alphaD, epsilon, maxStepsEpisode(i), savePath)
end

%% Pruebas epsilon
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas epsilon'
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 200; % Número de repeticiones de cada set de episodios
numEpisodes = 1;
alphaD = 0.0001;
maxStepsEpisode = 500;
epsilon = [0 0.001 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.4];
for i = 1:size(epsilon,2)
    RL_mount_buffer_function(numExperiments, numRep, numEpisodes, alphaD, epsilon(i), maxStepsEpisode, savePath)
end

%% Pruebas numRep
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas nRep'
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = [50 100 150 200]; % Número de repeticiones de cada set de episodios
numEpisodes = 1;
alphaD = 0.0001;
maxStepsEpisode = 500;
epsilon = 0.01;
for i = 2:size(numRep,2)
    RL_mount_buffer_function(numExperiments, numRep(i), numEpisodes, alphaD, epsilon, maxStepsEpisode, savePath)
end
