function r_total = LSPI_core (epsilon, K, num_episodes, max_steps, reset, env, theta)

num_episodes = 1;
% Get problem data
GetStateFeatures = env.GetStateFeatures;
DoAction = env.DoAction;
num_actions = env.num_actions;
M = env.M;
gamma = env.gamma;
r_total = zeros(1, num_episodes);
[Gamm, Lamb, z, t] = ResetLspiVal(M);

for n = 1:num_episodes
    
    % Initialize episode
    terminal = false;
    step = 0;
    s = env.initial_state;
    phi_s = GetStateFeatures(s, env);
    
    % Choose initial action
    a = EpsilonGreedy(epsilon, theta, phi_s, num_actions);
    
    % Run episode
    while ~terminal
        % env.PlotEpisode(s,a,step)
        % Build state-action features
        
        % Take action and observe state transition and reward
        [s_t1, r, terminal] = DoAction( a, s, env );
        phi_s_t1 = GetStateFeatures(s_t1, env);
        
        % Choose action for evaluating a possible future state-action pair
        a_t1 = EpsilonGreedy(epsilon, theta, phi_s_t1, num_actions);
        
        % Accumulate episode reward (assume gamma=1)
        r_total(n) = r_total(n) + gamma^step * r;
        
        
        % Move one step
        s = s_t1;
        a = a_t1;
        step = step + 1;
        t = t +1;
        
        
%         % Implicit policy improvement step
%         if ~mod(t, K)
%             theta = pinv(Gamm - Lamb)*z;
%         end
        
        % Terminate the episode
        if step == max_steps
            break
        end
        
    end
    
end
end
