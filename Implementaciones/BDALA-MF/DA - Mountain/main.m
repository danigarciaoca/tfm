clear all
close all
clc

C_W = ChainWalkSetUp();
reset = false; % flag to choose whether reset variables or not
num_episodes = 100; % number of episodes to be simulated
max_steps = 50; % maximum number of steps in one episode
K = 10;
epsilon = 0.1; % e-greedy parameter

GetStateFeatures = C_W.GetStateFeatures;
NA = C_W.N_actions; % number of actions
NS = C_W.N_states; % number of states
M = C_W.M; % state-action feature vector length
gamma = C_W.gamma; % discount factor

% Initialize variables
[Gamm, Lamb, z, t] = ResetLspiVal(M);
theta = rand(M,1); % parameter vector

for n = 1:num_episodes
    % Initialize episode
    terminal = false; % flag to indicate the end of the episode
    r_total = 0;
    step = 0; % step counter
    s = C_W.initial_state; % episode initial state
    
    phi_s = GetStateFeatures(s); % get state features
    a = EpsilonGreedy(theta, phi_s, epsilon, NA); % choose e-greedy action
    phi_sa = GetStateActionFeatures(phi_s, a, NA); % build state-action features
    
    while ~terminal
        
        % Take action and observe state transition and reward
        [s_t1, r] = getNextState(a, s, C_W);
        phi_s_t1 = GetStateFeatures(s_t1);
        
        % Choose action for evaluating a possible future state-action pair
        a_t1 = EpsilonGreedy(theta, phi_s_t1, epsilon, NA);
        phi_sa_t1 = GetStateActionFeatures(phi_s_t1, a_t1, NA);
        
        % Accumulate episode reward (assume gamma=1)
        r_total = r_total + gamma^step * r;
        
        % Build sample estimates
        Gamm = Gamm + phi_sa*phi_sa';
        Lamb = Lamb + phi_sa*phi_sa_t1';
        z = z + phi_sa*r;
        
        % Move one step
        s = s_t1;
        a = a_t1;
        phi_sa = phi_sa_t1;
        step = step + 1;
        t = t + 1;
        
        % Implicit policy improvement step
        if ~mod(t, K)
            theta = pinv(Gamm - gamma*Lamb)*z;
            if reset
                [Gamm, Lamb, z, t] = ResetLspiVal(M);
            end
        end
        
        % If next state is a terminal one or max_steps in an episode have been reached
        if step == max_steps
            terminal = true;
        end
    end
    
    % Display feedback
    fprintf('episode: %d, reward: %.3f\n', n, r_total)
end

% Compute theta
theta = pinv(Gamm - gamma*Lamb)*z;

% Get policy
policy = GetPolicy(theta, GetStateFeatures, NA, NS)
fprintf('1 = IZQUIERDA \n2 = DERECHA\n')