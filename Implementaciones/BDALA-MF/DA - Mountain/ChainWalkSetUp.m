function C_W = ChainWalkSetUp()
% Chain Walk problem

% Expected reward per state
Rs=[0 1 1 0]';
% Expected reward per state-action (s,a)
R = [0 1 0 1 1 0 1 0]';

% Probability transition matrix
P = [0.9, 0.1, 0, 0;
    0.1, 0.9, 0, 0;
    0.9, 0, 0.1, 0;
    0.1, 0, 0.9, 0;
    0, 0.9, 0, 0.1;
    0, 0.1, 0, 0.9;
    0, 0, 0.9, 0.1;
    0, 0, 0.1, 0.9];

N_states=4; % number of states
initial_state = 2; % initial state
final_state = [1 N_states];
% initial states distribution
mu = ((1/N_states)/(N_states-1))*ones(N_states,1); mu(initial_state) = 1-(1/N_states);

N_actions=2; % number of actions
gamma=.9; % discount factor

% Features
GetStateFeatures = @GetPolyFeatures; % state features
N_features = 3; % number of state features
M = N_features*N_actions; % number of state-action features

% Optimal policy
pi_opt = [0 1 0 0 0 0 0 0;
          0 0 0 1 0 0 0 0;
          0 0 0 0 1 0 0 0;
          0 0 0 0 0 0 1 0];

% Auxiliar matrix for constructing policy vector d
A = eye(N_states);
B = [1; 1];
duplicar = kron(A,B);

C_W = struct('N_states',N_states,'N_actions',N_actions,'P',P,'R',R,'Rs',Rs,...
    'gamma',gamma,'GetStateFeatures',GetStateFeatures,'N_features', N_features, ...
    'initial_state', initial_state, 'M', M, 'final_state', final_state, 'mu', mu,...
    'pi_opt', pi_opt, 'duplicar', duplicar)
end