% %% Pruebas alpha
% % clear all, close all, clc
% numExperiments = 30; % Número de experimentos de numRep*numEpisodes episodios
% numEpisodes = 6000;
% alpha = [1 0.7 0.5 0.3 0.1 0.07 0.05 0.03 0.01 0.005];
% epsilon = 0.2;
% for i = 1:size(alpha,2)
%     sarsa_func( numExperiments, numEpisodes, epsilon, alpha(i))
% end

%% Pruebas epsilon
% clear all, close all, clc
numExperiments = 30; % Número de experimentos de numRep*numEpisodes episodios
numEpisodes = 6000;
alpha = 0.1;
epsilon = [0 0.01 0.03 0.05 0.07 0.1 0.13 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.9];
for i = 1:size(epsilon,2)
    sarsa_func( numExperiments, numEpisodes, epsilon(i), alpha)
end