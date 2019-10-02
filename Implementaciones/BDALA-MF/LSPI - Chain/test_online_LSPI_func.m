function test_online_LSPI_func(problem, max_steps, num_episodes, num_exp, epsilon, rltv_path)


if strcmp(problem, 'ChainWalk')
    K = 10;
    env = GetChainWalkEnv();
    reset = 0;
    opt_policy = [0 1 0 1 1 0 1 0]; % Policy óptima en forma vector (2 acciones, 4 estados)
    errorD = nan(num_exp, num_episodes);
    
elseif strcmp(problem, 'MountainCar')
    max_steps = 500;
    num_episodes = 50;
    num_exp = 10;
    K = 100;
    N = 6;
    epsilon = 0.1;
    features = 'RBF';
    env = GetMountainCarEnv(features, N);
    reset = 0;

end

% OnlineLSPI
% Get problem data
GetStateFeatures = env.GetStateFeatures;
DoAction = env.DoAction;
num_actions = env.num_actions;
M = env.M;
gamma = env.gamma;
r_total = zeros(num_exp, num_episodes);
r_eps0_acumulada = zeros(num_exp, num_episodes);

for e = 1:num_exp
    e
    % Initialize algorithm
    [Gamm, Lamb, z, t] = ResetLspiVal(M);
    theta = rand(M,1);
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
            phi_sa = GetStateActionFeatures(phi_s, a, num_actions);
            
            % Take action and observe state transition and reward
            [s_t1, r, terminal] = DoAction( a, s, env );
            phi_s_t1 = GetStateFeatures(s_t1, env);
            
            % Choose action for evaluating a possible future state-action pair
            a_t1 = EpsilonGreedy(epsilon, theta, phi_s_t1, num_actions);
            phi_sa_t1 = GetStateActionFeatures(phi_s_t1, a_t1, num_actions);
            
            % Accumulate episode reward (assume gamma=1)
            r_total(e, n) = r_total(e, n) + gamma^step * r;
            
            % Build sample estimates
            Gamm = Gamm + phi_sa*phi_sa';
            Lamb = Lamb + phi_sa*phi_sa_t1';
            z = z + phi_sa*r;
            
            % Move one step
            s = s_t1;
            a = a_t1;
            phi_s = phi_s_t1;
            step = step + 1;
            t = t +1;
            
            
            % Implicit policy improvement step
            if ~mod(t, K)
                theta = pinv(Gamm - Lamb)*z;
                % theta(1:5)' % Para comprobar el orden de magnitud en el caso del Mountain car
            end
            
            % Terminate the episode
            if step == max_steps
                break
            end
            
        end
        if ~isfield(env, 'PlotEpisode') % En caso de que se trate del Mountain Car, evitamos esto ya que no hay policy exacta que podamos calcular.
            [policy, policy_vector] = GetPolicy(theta, env.GetStateFeatures, env.num_actions, env.S);
            errorD(e, n) = norm(abs(policy_vector - opt_policy),2);
        end
        
        if reset
            [Gamm, Lamb, z, t] = ResetLspiVal(M);
        end
        
        % Display feedback
        %fprintf('episode: %d, reward: %.3f\n', n, r_total(e, n))
    end
    G_eps0 = LSPI_core (0, K, num_episodes, max_steps, reset, env, theta);
    r_eps0_acumulada(e,:) = G_eps0;
end

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(r_total,1); % Media de los experimentos realizados
Gmean_eps0 = mean(r_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
mean_Gmean_eps0 = mean(Gmean_eps0); % Media de los episodios, promediados sobre los experimentos realizados, cuando epsilon = 0

if ~isfield(env, 'PlotEpisode') % Si se trata del Mountain Car, no corremos esto
    errorD_mean = mean(errorD,1); % Media de los experimentos realizados
    
    % EvalPolicy (theta, max_steps, env);
    [policy, policy_vector] = GetPolicy(theta, env.GetStateFeatures, env.num_actions, env.S);
end

save([rltv_path '\nExp=' num2str(num_exp) ',nEpi=' num2str(num_episodes) ',steps=' num2str(max_steps) ',eps=' num2str(epsilon) '.mat'])