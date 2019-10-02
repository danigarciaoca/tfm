% Online LSPI
% theta = lspi (epsilon, env, M)
% Inputs:
%   - epsilon: for epsilon greedy policy
%   - K: number of samples before estimating parameter theta
%   - num_episodes: maximum number of episodes
%   - max_steps: maximum number of steps in an episode
%   - reset: reset sample variables
%   - env: environment data
% Output:
%   - theta: state-action parameter, implies policy


function theta = OnlineLSPI (epsilon, K, num_episodes, max_steps, reset, env, num_exp)

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
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(r_total,1); % Media de los experimentos realizados
Gmean_eps0 = mean(r_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
figure, hold on
plot((1:num_episodes), Gmean, 'b', 'LineWidth', 2)
plot((1:num_episodes), Gmean_eps0, 'g', 'LineWidth', 2)
plot((1:num_episodes), mean(Gmean_eps0)*ones(size(Gmean_eps0)),'--r', 'LineWidth', 2)
hold off, title(['Return per episode (always starting at s=' ,num2str(env.initial_state), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon)], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0']},'Location','southeast','FontSize',12)

ylim([0 10])
end
