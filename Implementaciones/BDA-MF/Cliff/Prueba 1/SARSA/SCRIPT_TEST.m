clear all, close all, clc

numExperiments = 50; % Número de experimentos sobre el que se promedia
numEpisodes = 500; % Número de episodios de cada experimento
max_steps = 1000; % Máximo número de pasos por episodio

%% Pruebas alfa
rltv_path = 'Resultados SARSA\Pruebas alfa';
epsilon = 0.01;
alfa = [0.10 0.15 0.20 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1];
for i = 1:size(alfa,2)
    i
    sarsa_function(numExperiments, numEpisodes, max_steps, epsilon, alfa(i), rltv_path)
end

%% Pruebas epsilon
rltv_path = 'Resultados SARSA\Pruebas epsilon';
alfa = 0.5;
epsilon = [0.0001 0.0005 0.001 0.005 0.01 0.05 0.10 0.15 0.20];
for i = 1:size(epsilon,2)
    i
    sarsa_function(numExperiments, numEpisodes, max_steps, epsilon(i), alfa, rltv_path)
end