%% Pruebas alpha
clear all, close all, clc
numExperiments = 30; % Número de experimentos de numRep*numEpisodes episodios
numEpisodes = 2000;
alpha = [1 0.7 0.5 0.3 0.1 0.07 0.05 0.03 0.01 0.005];
epsilon = 0.1;
rltv_path = 'Resultados QL\Pruebas alfa';
for i = 1:size(alpha,2)
    i
    q_learning_func( numExperiments, numEpisodes, epsilon, alpha(i), rltv_path)
end

%% Pruebas epsilon
clear all, close all, clc
numExperiments = 30; % Número de experimentos de numRep*numEpisodes episodios
numEpisodes = 2000;
alpha = 0.3;
epsilon = [0 0.01 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.9];
rltv_path = 'Resultados QL\Pruebas epsilon';
for i = 1:size(epsilon,2)
    i
    q_learning_func( numExperiments, numEpisodes, epsilon(i), alpha, rltv_path)
end