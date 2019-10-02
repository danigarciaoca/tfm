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
num_dim = 2; % number of dimensions in the problem (2: speed and position)

if strcmp(feat, 'RBF')
    GetStateFeatures = @GetRBFFeatures;
end

% number of state-action features
M = N^num_dim*num_actions;

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

% Auxiliar matrix for constructing policy vector d
S_disc = N^num_dim; % discretized state space (useful in terms of getting a policy for each state in continuous state space problems)
A = eye(S_disc);
B = ones(num_actions,1);
duplicate = kron(A,B);

% Discretization of x (position) and y (speed) dimensions
mu_x = linspace(dimx(1), dimx(2), N+2);
mu_x = mu_x(2:end-1);
mu_y = linspace(dimy(1), dimy(2), N+2);
mu_y = mu_y(2:end-1);
xy_disc = combvec(mu_x,mu_y)'; % all possible combinations for future indexing of policy

% Build struct for passing parameters more easily
env.dimx = dimx;
env.dimy = dimy;
env.num_dim = num_dim;
env.GetStateFeatures = GetStateFeatures;
env.N = N;
env.S = S_disc; % discretization of continuous state space
env.mu_x = mu_x;
env.mu_y = mu_y;
env.xy_disc = xy_disc;

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

env.duplicate = duplicate;

env.PlotEpisode = PlotEpisode;

end