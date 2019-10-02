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
Pl = [0.9, 0.1, 0, 0;
      0.9, 0, 0.1, 0;
      0, 0.9, 0, 0.1;
      0, 0, 0.9, 0.1];
Pr = [0.1, 0.9, 0, 0;
      0.1, 0, 0.9, 0;
      0, 0.1, 0, 0.9;
      0, 0, 0.1, 0.9];
Pssa = nan(4,4,2);
Pssa(:,:,1) = Pl;
Pssa(:,:,2) = Pr;

Rs = [0, 1, 1, 0];

gamma= 0.9;




% Build struct for passing parameters more easily
env.S = S;

env.GetStateFeatures = GetStateFeatures;
env.initial_state = initial_state;

env.DoAction = DoAction;
env.actions_list = actions_list;
env.num_actions = num_actions;
env.M = M;

env.Pssa = Pssa;
env.Rs = Rs;

env.gamma = gamma;

end