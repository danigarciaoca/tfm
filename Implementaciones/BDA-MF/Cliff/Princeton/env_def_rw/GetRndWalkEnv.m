function env = GetRndWalkEnv(name, numStates, initType, transitionType, rewardType, finalReward)

% Random Walk problem
if strcmp(initType , 'center')
    initState = round((numStates+1)/2);
elseif strcmp(initType , 'leftCorner')
    initState = 2;
end
numActions=2;
terminal_states = [1 numStates];

gamma=.9;
v_ini=zeros(numStates,1);
q_ini=zeros(numStates*numActions,1);

R = zeros(numStates, 1);
if strcmp(rewardType, 'rand')
    statesWithReward = 1+randperm(numStates-2,round((numStates-2)/2)); % numStates-2 to select randomly from the 5 real possible states (i.e, without terminal states); +1 necessary offset to avoid terminal states
    %actionsWithReward = randi(2,size(statesWithReward));
    %s_a_reward_index = ((statesWithReward-1)*numActions)+actionsWithReward;
    % Asignamos a los estados con recompensa, una recompensa aleatoria
    % entre 1 y 2
    R(statesWithReward)=randi([1 2], round((numStates-2)/2),1);
    % Al mejor estado le asignamos más return, para que termine.
    R(end) = finalReward;
elseif strcmp(rewardType, 'det')
    R(end) = 1;
end
mu = (1/numStates)*ones(numStates,1); % initial distribution over states (pdf)

P = zeros(numStates*numActions, numStates);
P(1:2,1) = [1; 1]; % Transition to itself
P(end-1:end,end) = [1; 1]; % Transition to itself
for s = 2:numStates-1
    for a = 1:numActions
        if a == 1 % moverse a la derecha
            s_next_good = s+1;
            s_next_bad = s-1;
        elseif a == 2 % moverse a la izquierda
            s_next_good = s-1;
            s_next_bad = s+1;
        end
        
        if strcmp(transitionType, 'rand')
            min_p_good = 0.7;
            p_good = min_p_good+(1-min_p_good)*rand;
            p_bad = 1-p_good;
        elseif strcmp(transitionType, 'det')
            p_good = 1;
            p_bad = 1-p_good;
        end
        
        P((((s-1)*numActions)+a), s_next_good) = p_good;
        P((((s-1)*numActions)+a), s_next_bad) = p_bad;
    end
end

A = eye(numStates);
B = [1; 1];
duplicar = kron(A,B);

A = eye(numStates);
B = [1 0];
pi_opt = kron(A,B);

A = eye(numStates);
B = [1 1];
marg = kron(A,B);

DoAction = @DoActionRW;

env=struct('name', name, 'numStates',numStates,'numActions',numActions,'P',P,'R',R...
    ,'gamma',gamma,'pi_opt',pi_opt,'v_ini',v_ini,'q_ini',q_ini);

env.terminal_states = terminal_states;
env.initState = initState;
env.mu = mu;
env.duplicar = duplicar;
env.marg = marg;
env.DoAction = DoAction; % handler to DoAction function
end