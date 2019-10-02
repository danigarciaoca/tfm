clear all, close all, clc

% % % % ENVIRONMENT:
% addpath('env_def_cliff\', 'buffer\'), rmpath('env_def_rw\')
% name = 'Cliff';
% numFilTablero = 4;
% numColTablero = 12;
% rewards = [-1, -100]; % -1 en todas las transiciones, -100 si cae al cliff
% initType = 'fixed'; % fixed: empieza en la esquina inferior izquierda; random: empieza en cualquier estado
% transitionType = 'det';
% env = GetCliffEnv(name, numFilTablero, numColTablero, rewards, initType, transitionType);

addpath('env_def_rw\', 'buffer\'), rmpath('env_def_cliff\')
name = 'RandomWalk';
N_states = 13;
initType = 'leftCorner'; % {'center', 'leftCorner'}
transitionType = 'det'; % {'det', 'rand'}
rewardType = 'det'; % {'det', 'rand'} || if rand, extra argument: finalReward=50
env = GetRndWalkEnv(name, N_states, initType, transitionType, rewardType);

% Optimum value (value iteration)
N_steps = 1000;
[v_opt, q_opt, q_opt_format] = value_iteration(env, N_steps, env.terminal_states);
[~, pi_lineal] = max(q_opt_format,[],2);
if strcmp(env.name, 'Cliff')
    grid_result_vi = gridPolicyRepresentation(env, pi_lineal)
end

% % % % AGENTE:
numExperiments = 30; % Número de experimentos sobre el que se promedia
num_episodes = 100; % Número de episodios de cada experimento
max_steps = 1000;
G = zeros(numExperiments,num_episodes); % Return por episodio
G_eps0_acumulada = zeros(numExperiments,num_episodes); % Return por episodio cuando epsilon = 0

DoAction = env.DoAction; % sampling oracle (SO)
n = env.numStates; % number of states
m = env.numActions; % number of actions
gamma = env.gamma; % discount factor
sigma = max(env.R); % reward upper bound
e = ones(n,1); % vector with all entries equaling 1
xi = (sigma/sqrt(n))*e; % arbitrary vector with positive entries

% Initialize v and lambda
lim_sup_v = sigma/(1-gamma);
lim_sup_l = (norm(xi,1)*sigma)/(1-gamma);
BUFF_DIM = max_steps*num_episodes;
v = CreateCircularBuffer(BUFF_DIM, n);
lambda = CreateCircularBuffer(BUFF_DIM, [n m]);
v = AddItem(v, lim_sup_v.*rand(n,1));
lambda = AddItem(lambda, lim_sup_l.*rand(n,m));

% List of useful states (i.e: those which are not terminal)
[useful_states, useful_states_sz] = GetUsefulStates(n, env.terminal_states);
for e = 1:num_episodes
    step = 0; % step counter of each episode
    terminal = false;
    while ~terminal
        k = v.index; % lambda.index and v.index are always equal
        
        % Sample i uniformly from S
        i = useful_states(randperm(useful_states_sz,1));
        % Sample a uniformly from A
        a = randperm(m,1);
        % Sample j and r conditioned on (i,a) from SO
        [ j, r_ija, terminal ] = DoAction(a, i, env);
        % Set beta
        beta = sqrt(n/(k-1)); % k-1 because in the original paper v and lambda are indexed by 0; here by 1 [indexing v(0) in the paper is equivalent to indexing v(1) in Matlab, so in Matlab, k starts in k=2]
        
        % Update the primal iterates
        v_k = v.buffer(:, k-1);
        v_k(i) = max(min(v.buffer(i, k-1) - beta*((1/m)*xi(i) - lambda.buffer(i, a, k-1)), lim_sup_v),0);
        v_k(j) = max(min(v.buffer(j, k-1) - gamma*beta*lambda.buffer(i, a, k-1), lim_sup_v),0);
        
        % Update the dual iterates
        l_k = lambda.buffer(:, :, k-1);
        l_k(i,a) = lambda.buffer(i, a, k-1) + beta*(gamma*v.buffer(j, k-1) - v.buffer(i, k-1) + r_ija);
        % Project the dual iterates
        l_k = DualProjection_dMDP(l_k, xi, gamma, m, n);
        
        % Update values in buffer
        v = AddItem(v, v_k);
        lambda = AddItem(lambda, l_k);
        
        % Increment step count
        step = step + 1;
        if step == max_steps
            break;
        end
    end
end

lambda_hat = mean(lambda.buffer(:,:,1:lambda.index-1),3);
policy = lambda_hat./sum(lambda_hat,2)
[~, pi_lineal_max] = max(policy,[],2);