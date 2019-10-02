function game = random_walk_set_up_mod(N_states, transitionType, rewardType, finalReward)
% Random Walk problem
centralState = round((N_states+1)/2);
N_actions=2;
finalState = [1 N_states];

gamma=.9;
v_ini=zeros(N_states,1);
q_ini=zeros(N_states*N_actions,1);

R = zeros(N_states*N_actions, 1);
if strcmp(rewardType, 'rand')
    statesWithReward = 1+randperm(N_states-2,round((N_states-2)/2)); % N_states-2 to select randomly from the 5 real possible states (i.e, without terminal states); +1 necessary offset to avoid terminal states
    actionsWithReward = randi(2,size(statesWithReward));
    s_a_reward_index = ((statesWithReward-1)*N_actions)+actionsWithReward;
    % Asignamos a los estados con recompensa, una recompensa aleatoria
    % entre 1 y 2
    R(s_a_reward_index)=randi([1 2], round((N_states-2)/2),1);
    % Al mejor estado le asignamos más return, para que termine.
    R(end-3) = finalReward;
elseif strcmp(rewardType, 'det')
    R(end-3) = 1;
end
mu = (1/N_states)*ones(N_states,1); % initial distribution over states (pdf)

P = zeros(N_states*N_actions, N_states);
P(1:2,1) = [1; 1];
P(end-1:end,end) = [1; 1];
for s = 2:N_states-1
    for a = 1:N_actions
        if a == 1 % moverse a la derecha
            s_next_good = s+1;
            s_next_bad = s-1;
        elseif a == 2 % moverse a la izquierda
            s_next_good = s-1;
            s_next_bad = s+1;
        end
        
        if strcmp(transitionType, 'rand')
            min_p_good = 0.8;
            p_good = min_p_good+(1-min_p_good)*rand;
            p_bad = 1-p_good;
        elseif strcmp(transitionType, 'det')
            p_good = 1;
            p_bad = 1-p_good;
        end
        
        P((((s-1)*N_actions)+a), s_next_good) = p_good;
        P((((s-1)*N_actions)+a), s_next_bad) = p_bad;
    end
end

A = eye(N_states);
B = [1; 1];
duplicar = kron(A,B);

A = eye(N_states);
B = [1 0];
pi_opt = kron(A,B);

A = eye(N_states);
B = [1 1];
marg = kron(A,B);

game=struct('N_states',N_states,'N_actions',N_actions,'P',P,'R',R,'finalState',...
    finalState,'gamma',gamma,'pi_opt',pi_opt,'v_ini',v_ini,'q_ini',q_ini,...
    'centralState',centralState,'mu',mu,'duplicar',duplicar,'marg',marg)
end