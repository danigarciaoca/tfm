clearvars
clc
close all


problem = 'ChainWalk'; % { 'ChainWalk', 'MountainCar' }


if strcmp(problem, 'ChainWalk')
    max_steps = 50;
    num_episodes = 100;
    K = 10;
    epsilon = 0.1;
    env = GetChainWalkEnv();
    reset = 0;
    
elseif strcmp(problem, 'MountainCar')
    max_steps = 500;
    num_episodes = 50;
    K = 100;
    N = 6;
    epsilon = 0.1;
    features = 'RBF'; 
    env = GetMountainCarEnv(features, N);
    reset = 0;
    
end



theta = OnlineLSPI (epsilon, K, num_episodes, max_steps, reset, env);

% EvalPolicy (theta, max_steps, env);

policy = GetPolicy(theta, env.GetStateFeatures, env.num_actions, env.S)
fprintf('1 = IZQUIERDA \n2 = DERECHA\n')