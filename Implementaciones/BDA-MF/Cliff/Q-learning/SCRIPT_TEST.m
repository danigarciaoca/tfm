clear all, close all, clc

numExperiments = 100; % N�mero de experimentos sobre el que se promedia
numEpisodes = 500; % N�mero de episodios de cada experimento
max_steps = 10000; % M�ximo n�mero de pasos por episodio

%% Pruebas alfa
rltv_path = 'Resultados QL\Pruebas alfa';
epsilon = 0.01;
alfa = [0.10 0.15 0.20 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 1.1 1.2 1.3 1.4];
for i = 1:size(alfa,2)
    i
    q_learning_function(numExperiments, numEpisodes, max_steps, epsilon, alfa(i), rltv_path)
end

%% Pruebas epsilon
rltv_path = 'Resultados QL\Pruebas epsilon';
alfa = 1;
epsilon = [0.001 0.005 0.01 0.05 0.10 0.15 0.20];
for i = 1:size(epsilon,2)
    i
    q_learning_function(numExperiments, numEpisodes, max_steps, epsilon(i), alfa, rltv_path)
end