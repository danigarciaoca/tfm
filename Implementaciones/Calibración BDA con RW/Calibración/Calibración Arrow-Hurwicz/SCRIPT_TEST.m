% clear all
% close all
% clc

%% Pruebas numEpi
% clear all, close all, clc
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = [1 10 20 25 30 35 40 45 50 55 60];
alphaD = 0.2;
alphaTD = 0.2;
epsilon = 0.1;
for i = 1:size(numEpisodes,2)
    RL_func(numExperiments, numRep, numEpisodes(i), alphaD, alphaTD, epsilon)
end

%% Pruebas alphaD
% clear all, close all, clc
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 30;
alphaD = [1 0.8 0.6 0.5 0.4 0.2 0.1 0.08 0.05];
alphaTD = 0.2;
epsilon = 0.1;
for i = 1:size(alphaD,2)
    RL_func(numExperiments, numRep, numEpisodes, alphaD(i), alphaTD, epsilon)
end

%% Pruebas alphaTD
% clear all, close all, clc
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 30;
alphaD = 0.2;
alphaTD = [1 0.8 0.6 0.5 0.4 0.2 0.1 0.08 0.05];
epsilon = 0.1;
for i = 1:size(alphaTD,2)
    RL_func(numExperiments, numRep, numEpisodes, alphaD, alphaTD(i), epsilon)
end

%% Pruebas epsilon
% clear all, close all, clc
numExperiments = 50; % Número de experimentos de numRep*numEpisodes episodios
numRep = 50; % Número de repeticiones de cada set de episodios
numEpisodes = 30;
alphaD = 0.2;
alphaTD = 0.2;
epsilon = [0 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.7 0.9 1];
for i = 1:size(epsilon,2)
    RL_func(numExperiments, numRep, numEpisodes, alphaD, alphaTD, epsilon(i))
end