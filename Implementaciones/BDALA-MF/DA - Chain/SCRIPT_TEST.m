clear all, close all, clc

addpath('env_def', 'buffer')

%% Pruebas numEpi
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas nEpi';
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = [1 10 20 25 30 35 40 45 50 55 60];
alphaD = 0.2;
maxStepsEpisode = 50;
epsilon = 0.1;
for i = 1:size(numEpisodes,2)
    i
    RL_chain_buffer_function(numExperiments, numRep, numEpisodes(i), alphaD, epsilon, maxStepsEpisode, savePath)
end

%% Pruebas alphaD
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas alfaD';
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 10;
alphaD = [1 0.8 0.6 0.5 0.4 0.2 0.1 0.08 0.06 0.05 0.04 0.02];
maxStepsEpisode = 50;
epsilon = 0.1;
for i = 1:size(alphaD,2)
    RL_chain_buffer_function(numExperiments, numRep, numEpisodes, alphaD(i), epsilon, maxStepsEpisode, savePath)
end

%% Pruebas maxStepsEpisode
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas step';
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 10;
alphaD = 0.2;
maxStepsEpisode = [5 10 20 30 40 50 70 100 150 200];
epsilon = 0.1;
for i = 1:size(maxStepsEpisode,2)
    RL_chain_buffer_function(numExperiments, numRep, numEpisodes, alphaD, epsilon, maxStepsEpisode(i), savePath)
end

%% Pruebas epsilon
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas epsilon'
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 10;
alphaD = 0.2;
maxStepsEpisode = 50;
epsilon = [0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 0.9 1];
for i = 1:size(epsilon,2)
    RL_chain_buffer_function(numExperiments, numRep, numEpisodes, alphaD, epsilon(i), maxStepsEpisode, savePath)
end

%% Pruebas numRep
% clear all, close all, clc
savePath = 'Resultados DA\Pruebas nRep'
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = [1 10 20 25 30 35 40 45 50 55 60 65 70 80 90 100]; % Número de repeticiones de cada set de episodios
numEpisodes = 10;
alphaD = 0.2;
maxStepsEpisode = 50;
epsilon = 0.1;
for i = 1:size(numRep,2)
    RL_chain_buffer_function(numExperiments, numRep(i), numEpisodes, alphaD, epsilon, maxStepsEpisode, savePath)
end
