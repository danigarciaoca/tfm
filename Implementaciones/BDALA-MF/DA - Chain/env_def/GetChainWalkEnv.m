function env = GetChainWalkEnv()

% States
% -----------------
% bounds for position
S = 4;

% Actions
% -----------------
DoAction = @DoActionChainWalk;
actions_list = [-1 ; 1];
num_actions = length(actions_list);

% Features
% -----------------
% State features
GetStateFeatures = @GetPolyFeatures;
N = 3;
% number of state-action features
M = N*num_actions;

% Initial state
initial_state = 2;

% Transitions
Pssa = [0.9, 0.1, 0, 0; % s = 1, a = left (1)
        0.1, 0.9, 0, 0; % s = 1, a = right (2)
        0.9, 0, 0.1, 0; % s = 2, a = left (1)
        0.1, 0, 0.9, 0; % s = 2, a = right (2)
        0, 0.9, 0, 0.1; % s = 3, a = left (1)
        0, 0.1, 0, 0.9; % s = 3, a = right (2)
        0, 0, 0.9, 0.1; % s = 4, a = left (1)
        0, 0, 0.1, 0.9]; % s = 4, a = right (2)

Rs = [0, 1, 1, 0];

gamma= 0.9;

% Optimal policy
pi_opt = [0 1 0 0 0 0 0 0;
          0 0 0 1 0 0 0 0;
          0 0 0 0 1 0 0 0;
          0 0 0 0 0 0 1 0];
      
% Expected reward per state-action (s,a)
R = [0 1 0 1 1 0 1 0]';

% Optimum values
v_opt = inv(eye(S)-gamma*pi_opt*Pssa)*pi_opt*R;
q_opt = inv(eye(S*num_actions)-gamma*Pssa*pi_opt)*R;

% Auxiliar matrix for constructing policy vector d
A = eye(S);
B = ones(num_actions,1);
duplicate = kron(A,B);

% Build struct for passing parameters more easily
env.S = S;

env.GetStateFeatures = GetStateFeatures;
env.initial_state = initial_state;

env.DoAction = DoAction;
env.actions_list = actions_list;
env.num_actions = num_actions;
env.N = N;
env.M = M;

env.Pssa = Pssa;
env.Rs = Rs;

env.gamma = gamma;

env.v_opt = v_opt;
env.q_opt = q_opt;

env.duplicate = duplicate;
end