clear all, close all, clc

problem = 'ChainWalk'; % { 'ChainWalk', 'MountainCar' }

%% Pruebas max_steps
rltv_path = 'Resultados LSPI\Pruebas max_steps';
max_steps = [50 100 150 200 300];
num_episodes = 200;
num_exp = 50;
K = 10;
epsilon = 0.1;
for i = 1:size(max_steps,2)
    test_online_LSPI_func(problem, max_steps(i), num_episodes, num_exp, epsilon, rltv_path)
end

%% Pruebas num_episodes
rltv_path = 'Resultados LSPI\Pruebas num_episodes';
max_steps = 200;
num_episodes = [50 100 150 200 300];
num_exp = 50;
K = 10;
epsilon = 0.1;
for i = 1:size(num_episodes,2)
    test_online_LSPI_func(problem, max_steps, num_episodes(i), num_exp, epsilon, rltv_path)
end

%% Pruebas epsilon
rltv_path = 'Resultados LSPI\Pruebas epsilon';
max_steps = 200;
num_episodes = 200;
num_exp = 50;
K = 10;
epsilon = [0.05 0.1 0.15 0.2];
for i = 1:size(epsilon,2)
    test_online_LSPI_func(problem, max_steps, num_episodes, num_exp, epsilon(i), rltv_path)
end