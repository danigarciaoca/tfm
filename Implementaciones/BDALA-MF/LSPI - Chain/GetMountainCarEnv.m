function env = GetMountainCarEnv(feat, N)

% States
% -----------------
% bounds for position
bound_position_left = -1.2; 
bound_position_right = 0.6;

% bounds for speed
bound_speed_left = -0.07; 
bound_speed_right = 0.07;

% Actions
% -----------------
DoAction = @DoActionMountainCar;
actions_list = [-1 ; 0 ; 1];
num_actions = length(actions_list);

% Features
% -----------------
% State features
dimx = [bound_position_left, bound_position_right];
dimy = [bound_speed_left, bound_speed_right];

if strcmp(feat, 'RBF')
    GetStateFeatures = @GetRBFFeatures;
end

% number of state-action features
M = N^2*num_actions;

% Initial state
initial_position = -0.5;
initial_speed    =  0.0;
initial_state = [initial_position, initial_speed];

% Terminal state determined as position
goal = 0.6;

% Transitions
friction = 0.999;
time_step = 0.001;
slope = -0.0025;
slope_ampli = 3.0;


% Discount factor
gamma= 1;


% Plot during evaluation
PlotEpisode = @MountainCarPlot;


% Build struct for passing parameters more easily
env.dimx = dimx;
env.dimy = dimy;
env.GetStateFeatures = GetStateFeatures;
env.N = N;

env.bound_speed_left = bound_speed_left;
env.bound_speed_right = bound_speed_right;
env.bound_position_left = bound_position_left;
env.initial_state = initial_state;
env.goal = goal;

env.friction = friction;
env.time_step = time_step;
env.slope = slope;
env.slope_ampli = slope_ampli;

env.DoAction = DoAction;
env.actions_list = actions_list;
env.num_actions = num_actions;
env.M = M;

env.gamma = gamma;

env.PlotEpisode = PlotEpisode;

end