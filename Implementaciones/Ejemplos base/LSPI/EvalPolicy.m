function r_total = EvalPolicy (theta, max_steps, env)

GetStateFeatures = env.GetStateFeatures;
DoAction = env.DoAction;
num_actions = env.num_actions;
gamma = env.gamma;


% Initialize episode
terminal = false; 
r_total = 0;
step = 0;
s = env.initial_state; 
phi_s = GetStateFeatures(s, env);

a = Greedy(theta, phi_s, num_actions);

% Run episode
while ~terminal

    % Take action and observe state transition and reward
    [s_t1, r, terminal] = DoAction( a, s, env );
    phi_s_t1 = GetStateFeatures(s_t1, env);

    % Choose action for evaluating a possible future state-action pair
    a_t1 = Greedy(theta, phi_s_t1, num_actions);

    % Accumulate episode reward (assume gamma=1)
    r_total = r_total + gamma^step * r;

    if isfield(env, 'PlotEpisode')
        env.PlotEpisode(s, a, step);        
    end
    
    % Move one step
    s = s_t1;
    a = a_t1;
    step = step + 1;

    % Terminate the episode
    if step == max_steps 
        break
    end

end

fprintf('step: %d, reward: %.3f\n', step, r_total)

end